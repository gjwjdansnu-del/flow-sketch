import * as ort from 'onnxruntime-web'
import {
  MASK_COLS,
  MASK_ROWS,
  type FieldKey,
  type PredictResponse,
} from './constants'

const OUTPUT_KEYS: FieldKey[] = [
  'mach',
  'pressure',
  'density',
  'temperature',
  'shock_indicator',
]

export type ModelLoadStatus = 'idle' | 'loading' | 'loaded' | 'error'

type OutputStats = {
  mean: number[]
  std: number[]
  keys: string[]
}

const modelUrl = `${import.meta.env.BASE_URL}models/unet_site.onnx`
const statsUrl = `${import.meta.env.BASE_URL}models/unet_site_stats.json`

let session: ort.InferenceSession | null = null
let outputStats: OutputStats | null = null
let loadPromise: Promise<void> | null = null
let loadStatus: ModelLoadStatus = 'idle'
let loadError: string | null = null

// Match package.json onnxruntime-web version for WASM fallback assets.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'

function executionProviders(): ort.InferenceSession.ExecutionProviderConfig[] {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    return ['webgpu', 'wasm']
  }
  return ['wasm']
}

export function getModelLoadStatus(): ModelLoadStatus {
  return loadStatus
}

export function getModelLoadError(): string | null {
  return loadError
}

export function isModelLoaded(): boolean {
  return loadStatus === 'loaded' && session !== null && outputStats !== null
}

export async function ensureModelLoaded(): Promise<void> {
  if (loadStatus === 'loaded' && session && outputStats) {
    return
  }
  if (loadStatus === 'error') {
    throw new Error(
      loadError ??
        'Browser model failed to load. Please refresh or check network.',
    )
  }
  if (loadPromise) {
    return loadPromise
  }

  loadStatus = 'loading'
  loadError = null

  loadPromise = (async () => {
    try {
      const statsResponse = await fetch(statsUrl)
      if (!statsResponse.ok) {
        throw new Error(`Failed to load stats (${statsResponse.status})`)
      }
      outputStats = (await statsResponse.json()) as OutputStats

      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: executionProviders(),
      })
      loadStatus = 'loaded'
    } catch (error) {
      loadStatus = 'error'
      loadError =
        error instanceof Error
          ? error.message
          : 'Browser model failed to load. Please refresh or check network.'
      session = null
      outputStats = null
      throw new Error(
        'Browser model failed to load. Please refresh or check network.',
      )
    } finally {
      loadPromise = null
    }
  })()

  return loadPromise
}

function buildInputTensor(
  solidMask: number[][],
  mach: number,
): ort.Tensor {
  const ny = MASK_ROWS
  const nx = MASK_COLS
  const planeSize = ny * nx
  const data = new Float32Array(2 * planeSize)

  for (let row = 0; row < ny; row += 1) {
    const maskRow = solidMask[row]
    if (!maskRow || maskRow.length !== nx) {
      throw new Error(`solid_mask must have shape (${ny}, ${nx}).`)
    }
    for (let col = 0; col < nx; col += 1) {
      const idx = row * nx + col
      data[idx] = maskRow[col]
      data[planeSize + idx] = mach
    }
  }

  return new ort.Tensor('float32', data, [1, 2, ny, nx])
}

function tensorToFields(output: ort.Tensor, stats: OutputStats): PredictResponse {
  const ny = MASK_ROWS
  const nx = MASK_COLS
  const values = output.data as Float32Array
  const planeSize = ny * nx
  const result = {} as PredictResponse

  for (let channel = 0; channel < OUTPUT_KEYS.length; channel += 1) {
    const key = OUTPUT_KEYS[channel]
    const mean = stats.mean[channel] ?? 0
    const std = stats.std[channel] ?? 1
    const offset = channel * planeSize
    const field: number[][] = []

    for (let row = 0; row < ny; row += 1) {
      const rowValues: number[] = []
      for (let col = 0; col < nx; col += 1) {
        const normalized = values[offset + row * nx + col]
        rowValues.push(normalized * std + mean)
      }
      field.push(rowValues)
    }
    result[key] = field
  }

  return result
}

export async function predictFlowInBrowser(
  solidMask: number[][],
  mach: number,
): Promise<PredictResponse> {
  await ensureModelLoaded()
  if (!session || !outputStats) {
    throw new Error('Model session is not available.')
  }

  const input = buildInputTensor(solidMask, mach)
  try {
    const results = await session.run({ input })
    const output = results.output
    if (!output) {
      throw new Error('ONNX session returned no output tensor.')
    }
    return tensorToFields(output, outputStats)
  } finally {
    input.dispose()
  }
}
