import {
  DISPLAY_HEIGHT,
  DISPLAY_WIDTH,
  GRID_HEIGHT,
  GRID_WIDTH,
  MASK_COLS,
  MASK_ROWS,
  X_MAX,
  X_MIN,
  Y_MAX,
  Y_MIN,
} from './constants'

export type Point = { x: number; y: number }

export function canvasToPhysical(canvasX: number, canvasY: number): Point {
  const x = X_MIN + (canvasX / DISPLAY_WIDTH) * (X_MAX - X_MIN)
  const y = Y_MAX - (canvasY / DISPLAY_HEIGHT) * (Y_MAX - Y_MIN)
  return { x, y }
}

export function physicalToCanvas(physicalX: number, physicalY: number): Point {
  const x = ((physicalX - X_MIN) / (X_MAX - X_MIN)) * DISPLAY_WIDTH
  const y = ((Y_MAX - physicalY) / (Y_MAX - Y_MIN)) * DISPLAY_HEIGHT
  return { x, y }
}

/**
 * Map canvas pixel y (origin top, y down) to mask row index.
 * Mask row 0 = physical y_min; row MASK_ROWS-1 = physical y_max.
 * Matches canvasToPhysical / physicalToCanvas convention.
 */
export function canvasPixelYToMaskRow(py: number): number {
  const t = py / Math.max(DISPLAY_HEIGHT - 1, 1)
  const row = Math.round((1 - t) * (MASK_ROWS - 1))
  return Math.max(0, Math.min(MASK_ROWS - 1, row))
}

export function canvasPixelXToMaskCol(px: number): number {
  const t = px / Math.max(DISPLAY_WIDTH - 1, 1)
  const col = Math.round(t * (MASK_COLS - 1))
  return Math.max(0, Math.min(MASK_COLS - 1, col))
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y
    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + Number.EPSILON) + xi
    if (intersects) {
      inside = !inside
    }
  }
  return inside
}

export function polygonToSolidMask(polygon: Point[]): number[][] {
  const mask = Array.from({ length: MASK_ROWS }, () =>
    Array.from({ length: MASK_COLS }, () => 0),
  )

  if (polygon.length < 3) {
    return mask
  }

  for (let row = 0; row < MASK_ROWS; row++) {
    const y =
      Y_MIN + (row / Math.max(MASK_ROWS - 1, 1)) * (Y_MAX - Y_MIN)
    for (let col = 0; col < MASK_COLS; col++) {
      const x =
        X_MIN + (col / Math.max(MASK_COLS - 1, 1)) * (X_MAX - X_MIN)
      if (pointInPolygon({ x, y }, polygon)) {
        mask[row][col] = 1
      }
    }
  }

  return mask
}

export function distance(a: Point, b: Point): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.hypot(dx, dy)
}

export function isNearStart(
  point: Point,
  start: Point,
  threshold = 12,
): boolean {
  return distance(point, start) <= threshold
}

export function drawPolygonPreview(
  ctx: CanvasRenderingContext2D,
  polygonCanvas: Point[],
  cursor?: Point | null,
  closed = false,
): void {
  ctx.clearRect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)
  ctx.fillStyle = '#f4f4f4'
  ctx.fillRect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)

  ctx.strokeStyle = '#888'
  ctx.lineWidth = 1
  for (let x = 0; x <= GRID_WIDTH; x += GRID_WIDTH / 4) {
    const px = (x / GRID_WIDTH) * DISPLAY_WIDTH
    ctx.beginPath()
    ctx.moveTo(px, 0)
    ctx.lineTo(px, DISPLAY_HEIGHT)
    ctx.stroke()
  }
  for (let y = 0; y <= GRID_HEIGHT; y += GRID_HEIGHT / 4) {
    const py = (y / GRID_HEIGHT) * DISPLAY_HEIGHT
    ctx.beginPath()
    ctx.moveTo(0, py)
    ctx.lineTo(DISPLAY_WIDTH, py)
    ctx.stroke()
  }

  if (polygonCanvas.length === 0) {
    return
  }

  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
  ctx.strokeStyle = '#111'
  ctx.lineWidth = 2

  if (closed && polygonCanvas.length >= 3) {
    ctx.beginPath()
    ctx.moveTo(polygonCanvas[0].x, polygonCanvas[0].y)
    for (let i = 1; i < polygonCanvas.length; i++) {
      ctx.lineTo(polygonCanvas[i].x, polygonCanvas[i].y)
    }
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
  } else {
    ctx.beginPath()
    ctx.moveTo(polygonCanvas[0].x, polygonCanvas[0].y)
    for (let i = 1; i < polygonCanvas.length; i++) {
      ctx.lineTo(polygonCanvas[i].x, polygonCanvas[i].y)
    }
    if (cursor) {
      ctx.lineTo(cursor.x, cursor.y)
    }
    ctx.stroke()
  }

  for (const point of polygonCanvas) {
    ctx.beginPath()
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2)
    ctx.fillStyle = closed ? '#fff' : '#111'
    ctx.fill()
    ctx.strokeStyle = '#111'
    ctx.stroke()
  }
}
