import { DISPLAY_HEIGHT, DISPLAY_WIDTH } from './constants'
import { canvasPixelXToMaskCol, canvasPixelYToMaskRow } from './mask'

export const COLORBAR_WIDTH = 72
export const COLORBAR_HEIGHT = DISPLAY_HEIGHT

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function turboColor(t: number): [number, number, number] {
  const x = clamp01(t)
  const r = Math.round(255 * clamp01(1.5 * x - 0.25))
  const g = Math.round(255 * clamp01(1.2 - Math.abs(x - 0.5) * 2.2))
  const b = Math.round(255 * clamp01(1.3 - x * 1.4))
  return [r, g, b]
}

export function fieldMinMax(field: number[][]): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const row of field) {
    for (const value of row) {
      if (!Number.isFinite(value)) {
        continue
      }
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { min: 0, max: 1 }
  }
  return { min, max }
}

export function drawFieldColormap(
  ctx: CanvasRenderingContext2D,
  field: number[][],
  solidMask: number[][],
  vmin: number,
  vmax: number,
): void {
  const image = ctx.createImageData(DISPLAY_WIDTH, DISPLAY_HEIGHT)

  for (let py = 0; py < DISPLAY_HEIGHT; py++) {
    const row = canvasPixelYToMaskRow(py)
    for (let px = 0; px < DISPLAY_WIDTH; px++) {
      const col = canvasPixelXToMaskCol(px)
      const solid = solidMask[row]?.[col] ?? 0
      const value = field[row]?.[col] ?? 0
      const t = (value - vmin) / (vmax - vmin || 1)
      const [r, g, b] = solid > 0.5 ? [0, 0, 0] : turboColor(t)
      const offset = (py * DISPLAY_WIDTH + px) * 4
      image.data[offset] = r
      image.data[offset + 1] = g
      image.data[offset + 2] = b
      image.data[offset + 3] = 255
    }
  }

  ctx.putImageData(image, 0, 0)
}

export function formatFieldValue(value: number): string {
  if (!Number.isFinite(value)) {
    return '—'
  }
  const abs = Math.abs(value)
  if (abs >= 10_000 || (abs > 0 && abs < 0.001)) {
    return value.toExponential(2)
  }
  if (abs >= 1000) {
    return value.toFixed(0)
  }
  if (abs >= 10) {
    return value.toFixed(1)
  }
  return value.toFixed(2)
}

export function drawColorbar(
  ctx: CanvasRenderingContext2D,
  vmin: number,
  vmax: number,
  width: number = COLORBAR_WIDTH,
  height: number = COLORBAR_HEIGHT,
): void {
  ctx.clearRect(0, 0, width, height)

  const barX = 8
  const barWidth = 18
  const labelX = barX + barWidth + 6

  for (let py = 0; py < height; py++) {
    const t = 1 - py / Math.max(height - 1, 1)
    const [r, g, b] = turboColor(t)
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
    ctx.fillRect(barX, py, barWidth, 1)
  }

  ctx.strokeStyle = '#333'
  ctx.lineWidth = 1
  ctx.strokeRect(barX + 0.5, 0.5, barWidth - 1, height - 1)

  ctx.fillStyle = '#111'
  ctx.font = '11px system-ui, sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'top'
  ctx.fillText(formatFieldValue(vmax), labelX, 4)
  ctx.textBaseline = 'bottom'
  ctx.fillText(formatFieldValue(vmin), labelX, height - 4)

  const mid = (vmin + vmax) / 2
  ctx.textBaseline = 'middle'
  ctx.fillText(formatFieldValue(mid), labelX, height / 2)
}
