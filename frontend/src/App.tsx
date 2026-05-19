import { useCallback, useEffect, useRef, useState } from 'react'
import { predictFlow } from './api'
import { getModelLoadStatus } from './lib/onnxInference'
import {
  FIELD_OPTIONS,
  type FieldKey,
  type PredictResponse,
} from './lib/constants'
import {
  COLORBAR_HEIGHT,
  COLORBAR_WIDTH,
  drawColorbar,
  drawFieldColormap,
  fieldMinMax,
} from './lib/colormap'
import {
  canvasToPhysical,
  drawPolygonPreview,
  isNearStart,
  type Point,
  physicalToCanvas,
  polygonToSolidMask,
} from './lib/mask'
import { normalizeGeometry, rotateGeometry } from './lib/normalize'
import { getPresetPolygon, PRESET_OPTIONS, type PresetId } from './lib/presets'
import './App.css'

const DEBOUNCE_MS = 150

const WORKFLOW_STEPS = [
  'Draw a 2D body shape, or choose a preset.',
  'Click Predict.',
  'Select Mach, Pressure, Density, Temperature, or Shock to inspect the predicted field.',
  'Adjust Mach number and AoA to observe real-time changes.',
] as const

function App() {
  const drawCanvasRef = useRef<HTMLCanvasElement>(null)
  const resultCanvasRef = useRef<HTMLCanvasElement>(null)
  const colorbarCanvasRef = useRef<HTMLCanvasElement>(null)
  const solidMaskRef = useRef<number[][] | null>(null)
  const normalizedPolygonRef = useRef<Point[] | null>(null)
  const predictRequestIdRef = useRef(0)
  const autoUpdateEnabledRef = useRef(false)

  const [polygonCanvas, setPolygonCanvas] = useState<Point[]>([])
  const [polygonClosed, setPolygonClosed] = useState(false)
  const [cursor, setCursor] = useState<Point | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const [mach, setMach] = useState(2.0)
  const [aoa, setAoa] = useState(0)
  const [selectedField, setSelectedField] = useState<FieldKey>('mach')

  const [solidMask, setSolidMask] = useState<number[][] | null>(null)
  const [prediction, setPrediction] = useState<PredictResponse | null>(null)
  const [inFlight, setInFlight] = useState(false)
  const [autoUpdateEnabled, setAutoUpdateEnabled] = useState(false)
  const [shapeDirty, setShapeDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelStatus, setModelStatus] = useState<
    'idle' | 'loading' | 'loaded' | 'error'
  >('idle')

  const disableAutoUpdate = useCallback(() => {
    autoUpdateEnabledRef.current = false
    setAutoUpdateEnabled(false)
  }, [])

  const markShapeEdited = useCallback(() => {
    setShapeDirty(true)
    disableAutoUpdate()
  }, [disableAutoUpdate])

  const redrawDrawCanvas = useCallback(() => {
    const canvas = drawCanvasRef.current
    if (!canvas) {
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }
    drawPolygonPreview(ctx, polygonCanvas, polygonClosed ? null : cursor, polygonClosed)
  }, [polygonCanvas, polygonClosed, cursor])

  const redrawResultCanvas = useCallback(() => {
    const canvas = resultCanvasRef.current
    const colorbarCanvas = colorbarCanvasRef.current
    if (!canvas || !colorbarCanvas) {
      return
    }

    const fieldCtx = canvas.getContext('2d')
    const colorbarCtx = colorbarCanvas.getContext('2d')
    if (!fieldCtx || !colorbarCtx) {
      return
    }

    if (!prediction || !solidMask) {
      fieldCtx.clearRect(0, 0, canvas.width, canvas.height)
      colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height)
      return
    }

    const field = prediction[selectedField]
    const { min, max } = fieldMinMax(field)
    drawFieldColormap(fieldCtx, field, solidMask, min, max)
    drawColorbar(colorbarCtx, min, max)
  }, [prediction, solidMask, selectedField])

  const maskFromNormalizedPolygon = useCallback(
    (normalized: Point[], aoaValue: number): number[][] => {
      const rotated = rotateGeometry(normalized, aoaValue)
      return polygonToSolidMask(rotated)
    },
    [],
  )

  const runPredict = useCallback(
    async (
      mask: number[][],
      machValue: number,
      options?: { enableAutoAfter?: boolean },
    ) => {
      const requestId = ++predictRequestIdRef.current
      const wasLoaded = getModelLoadStatus() === 'loaded'
      setInFlight(true)
      setError(null)
      if (!wasLoaded) {
        setModelStatus('loading')
      }

      try {
        const response = await predictFlow({
          solid_mask: mask,
          mach: machValue,
        })

        if (requestId !== predictRequestIdRef.current) {
          return
        }

        setPrediction(response)
        setSolidMask(mask)
        solidMaskRef.current = mask
        setModelStatus('loaded')

        if (options?.enableAutoAfter) {
          autoUpdateEnabledRef.current = true
          setAutoUpdateEnabled(true)
          setShapeDirty(false)
        }
      } catch (predictError) {
        if (requestId !== predictRequestIdRef.current) {
          return
        }
        const message =
          predictError instanceof Error
            ? predictError.message
            : 'Prediction failed.'
        setError(message)
        if (getModelLoadStatus() === 'error') {
          setModelStatus('error')
        }
      } finally {
        if (requestId === predictRequestIdRef.current) {
          setInFlight(false)
        }
      }
    },
    [],
  )

  const buildMaskFromPolygon = useCallback(
    (aoaValue: number): number[][] | null => {
      if (polygonCanvas.length < 3) {
        return null
      }
      try {
        const physical = polygonCanvas.map((point) =>
          canvasToPhysical(point.x, point.y),
        )
        const normalized = normalizeGeometry(physical)
        normalizedPolygonRef.current = normalized
        return maskFromNormalizedPolygon(normalized, aoaValue)
      } catch (normalizeError) {
        const message =
          normalizeError instanceof Error
            ? normalizeError.message
            : 'Failed to normalize drawn shape.'
        setError(message)
        return null
      }
    },
    [polygonCanvas, maskFromNormalizedPolygon],
  )

  useEffect(() => {
    redrawDrawCanvas()
  }, [redrawDrawCanvas])

  useEffect(() => {
    redrawResultCanvas()
  }, [redrawResultCanvas])

  useEffect(() => {
    if (!autoUpdateEnabledRef.current || !normalizedPolygonRef.current) {
      return
    }

    const timer = window.setTimeout(() => {
      try {
        const mask = maskFromNormalizedPolygon(normalizedPolygonRef.current!, aoa)
        void runPredict(mask, mach)
      } catch (rotateError) {
        const message =
          rotateError instanceof Error
            ? rotateError.message
            : 'Failed to rotate shape for AoA.'
        setError(message)
      }
    }, DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [mach, aoa, runPredict, maskFromNormalizedPolygon])

  const closePolygon = useCallback(
    (points: Point[]) => {
      if (points.length < 3) {
        return
      }
      setPolygonCanvas(points)
      setPolygonClosed(true)
      setSolidMask(null)
    },
    [],
  )

  const addPoint = useCallback(
    (point: Point) => {
      if (polygonClosed) {
        return
      }
      markShapeEdited()
      setPolygonCanvas((current) => {
        const last = current[current.length - 1]
        if (last && Math.hypot(last.x - point.x, last.y - point.y) < 4) {
          return current
        }
        return [...current, point]
      })
    },
    [polygonClosed, markShapeEdited],
  )

  const canvasPoint = (event: React.MouseEvent<HTMLCanvasElement>): Point => {
    const canvas = drawCanvasRef.current
    if (!canvas) {
      return { x: 0, y: 0 }
    }
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    }
  }

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event)

    if (polygonClosed) {
      markShapeEdited()
      setPolygonClosed(false)
      setPolygonCanvas([point])
      setIsDragging(true)
      return
    }

    setIsDragging(true)
    addPoint(point)
  }

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event)
    if (!polygonClosed) {
      setCursor(point)
    }
    if (isDragging && !polygonClosed) {
      addPoint(point)
    }
  }

  const handleMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const point = canvasPoint(event)
    if (
      !polygonClosed &&
      polygonCanvas.length >= 3 &&
      isNearStart(point, polygonCanvas[0])
    ) {
      closePolygon(polygonCanvas)
    }
    setIsDragging(false)
  }

  const handleMouseLeave = () => {
    setIsDragging(false)
    setCursor(null)
  }

  const handleDoubleClick = () => {
    if (!polygonClosed && polygonCanvas.length >= 3) {
      closePolygon(polygonCanvas)
    }
  }

  const handleLoadPreset = (presetId: PresetId) => {
    const physicalPolygon = getPresetPolygon(presetId)
    const canvasPolygon = physicalPolygon.map((point) =>
      physicalToCanvas(point.x, point.y),
    )

    disableAutoUpdate()
    solidMaskRef.current = null
    normalizedPolygonRef.current = null

    setPolygonCanvas(canvasPolygon)
    setPolygonClosed(true)
    setCursor(null)
    setIsDragging(false)
    setSolidMask(null)
    setShapeDirty(true)
    setError(null)
  }

  const handleClear = () => {
    predictRequestIdRef.current += 1
    autoUpdateEnabledRef.current = false
    solidMaskRef.current = null
    normalizedPolygonRef.current = null

    setPolygonCanvas([])
    setPolygonClosed(false)
    setCursor(null)
    setSolidMask(null)
    setPrediction(null)
    setAutoUpdateEnabled(false)
    setShapeDirty(false)
    setInFlight(false)
    setError(null)

    const resultCanvas = resultCanvasRef.current
    const colorbarCanvas = colorbarCanvasRef.current
    const fieldCtx = resultCanvas?.getContext('2d')
    const colorbarCtx = colorbarCanvas?.getContext('2d')
    if (fieldCtx && resultCanvas) {
      fieldCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height)
    }
    if (colorbarCtx && colorbarCanvas) {
      colorbarCtx.clearRect(0, 0, colorbarCanvas.width, colorbarCanvas.height)
    }
  }

  const handlePredict = async () => {
    let mask = solidMaskRef.current

    if (shapeDirty || !normalizedPolygonRef.current) {
      mask = buildMaskFromPolygon(aoa)
      if (!mask) {
        setError('Draw a closed body with at least 3 points.')
        return
      }
      setPolygonClosed(true)
    } else {
      try {
        mask = maskFromNormalizedPolygon(normalizedPolygonRef.current, aoa)
      } catch (rotateError) {
        const message =
          rotateError instanceof Error
            ? rotateError.message
            : 'Failed to rotate shape for AoA.'
        setError(message)
        return
      }
    }

    await runPredict(mask, mach, { enableAutoAfter: true })
  }

  const modelNotice =
    modelStatus === 'error' ? (
      <p className="model-notice model-notice--error" role="status">
        Browser model failed to load. Please refresh or check network.
      </p>
    ) : modelStatus === 'loading' ? (
      <p className="model-notice" role="status">
        Downloading AI model…
      </p>
    ) : modelStatus === 'loaded' ? (
      <p className="model-notice" role="status">
        Model loaded locally.
      </p>
    ) : (
      <p className="model-notice" role="status">
        First prediction may take a few seconds while the AI model downloads.
        After that, predictions run locally in your browser.
      </p>
    )

  return (
    <div className="app">
      <header className="site-header">
        <h1>
          DrawSupersonic
          <span className="version">ver 1</span>
        </h1>
        <p className="subtitle">
          Draw a 2D body and preview a real-time AI flow surrogate.
        </p>
        <ol className="workflow-guide" aria-label="How to use DrawSupersonic">
          {WORKFLOW_STEPS.map((text, index) => (
            <li key={text}>
              <span className="workflow-step-label">Step {index + 1}.</span> {text}
            </li>
          ))}
        </ol>
      </header>

      <section className="section-card shape-section" aria-label="Body shape">
        <div className="section-heading">
          <h2>Body shape</h2>
          <p className="section-hint">
            Click or drag points; double-click or click near the start point to close.
          </p>
        </div>
        <div className="presets">
          <span className="presets-label">Presets</span>
          <div className="preset-buttons" role="group" aria-label="Shape presets">
            {PRESET_OPTIONS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className="preset-button"
                onClick={() => handleLoadPreset(preset.id)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <div className="canvas-stage">
          <canvas
            ref={drawCanvasRef}
            width={512}
            height={256}
            className="canvas"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onDoubleClick={handleDoubleClick}
          />
        </div>
        <div className="shape-actions">
          <button type="button" className="secondary-button" onClick={handleClear}>
            Clear
          </button>
        </div>
      </section>

      <section
        className="section-card predict-section"
        aria-label="Prediction controls"
      >
        <div className="section-heading">
          <h2>Predict flow</h2>
        </div>
        <div className="predict-row">
          <button
            type="button"
            className="primary-button"
            onClick={handlePredict}
            disabled={inFlight && !autoUpdateEnabled}
          >
            {inFlight && !autoUpdateEnabled
              ? modelStatus === 'loading'
                ? 'Loading model...'
                : 'Predicting...'
              : 'Predict'}
          </button>
          {modelNotice}
        </div>
        {(autoUpdateEnabled || (shapeDirty && prediction)) && (
          <p className="session-hint">
            {autoUpdateEnabled &&
              'Mach and AoA update the field automatically after the first predict.'}
            {shapeDirty &&
              prediction &&
              ' Shape changed — click Predict to refresh.'}
          </p>
        )}
        <div className="sliders-row">
          <label>
            Mach: {mach.toFixed(1)}
            <input
              type="range"
              min={1.5}
              max={5}
              step={0.1}
              value={mach}
              onChange={(event) => setMach(Number(event.target.value))}
            />
          </label>
          <label>
            AoA: {aoa}°
            <input
              type="range"
              min={-10}
              max={10}
              step={1}
              value={aoa}
              onChange={(event) => setAoa(Number(event.target.value))}
            />
          </label>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section
        className="section-card field-section"
        aria-label="Predicted field visualization"
      >
        <div className="section-heading">
          <h2>Predicted field</h2>
          <p className="section-hint">
            Scalar field with body overlay in rotated coordinates.
          </p>
        </div>
        <div className="field-buttons" role="group" aria-label="Field selector">
          {FIELD_OPTIONS.map((option) => (
            <button
              key={option.key}
              type="button"
              className={selectedField === option.key ? 'active' : ''}
              onClick={() => setSelectedField(option.key)}
              disabled={!prediction}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="canvas-stage field-stage">
          <canvas
            ref={resultCanvasRef}
            width={512}
            height={256}
            className="canvas field-canvas"
          />
          <canvas
            ref={colorbarCanvasRef}
            width={COLORBAR_WIDTH}
            height={COLORBAR_HEIGHT}
            className="colorbar-canvas"
            aria-label="Field value colorbar"
          />
        </div>
      </section>

      <footer className="site-footer">
        <p>
          Experimental AI surrogate — not a validated CFD solver. Trained on SU2 Euler
          simulations (Mach 1.5–5, ±10° AoA). Browser-side U-Net inference; AoA is
          applied by rotating the body mask.
        </p>
      </footer>
    </div>
  )
}

export default App
