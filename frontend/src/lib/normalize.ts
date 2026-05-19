/** Geometry normalization for AI input (matches backend CFD convention). */

import type { Point } from './mask'

export const TARGET_CHORD = 1.0
export const DOMAIN_X_MIN = -1
export const DOMAIN_X_MAX = 3
export const DOMAIN_Y_MIN = -1
export const DOMAIN_Y_MAX = 1
export const DOMAIN_MARGIN = 0.05

function openLoop(points: Point[]): Point[] {
  if (
    points.length > 1 &&
    Math.hypot(points[0].x - points.at(-1)!.x, points[0].y - points.at(-1)!.y) <
      1e-9
  ) {
    return points.slice(0, -1)
  }
  return [...points]
}

export function polygonArea(points: Point[]): number {
  const open = openLoop(points)
  if (open.length < 3) {
    return 0
  }
  let area = 0
  for (let i = 0; i < open.length; i++) {
    const j = (i + 1) % open.length
    area += open[i].x * open[j].y - open[j].x * open[i].y
  }
  return area / 2
}

export function polygonCentroid(points: Point[]): Point {
  const open = openLoop(points)
  if (open.length === 0) {
    return { x: 0, y: 0 }
  }
  if (open.length < 3) {
    const meanX = open.reduce((sum, p) => sum + p.x, 0) / open.length
    const meanY = open.reduce((sum, p) => sum + p.y, 0) / open.length
    return { x: meanX, y: meanY }
  }

  let crossSum = 0
  let cx = 0
  let cy = 0
  for (let i = 0; i < open.length; i++) {
    const j = (i + 1) % open.length
    const cross = open[i].x * open[j].y - open[j].x * open[i].y
    crossSum += cross
    cx += (open[i].x + open[j].x) * cross
    cy += (open[i].y + open[j].y) * cross
  }

  if (Math.abs(crossSum) < 1e-12) {
    const meanX = open.reduce((sum, p) => sum + p.x, 0) / open.length
    const meanY = open.reduce((sum, p) => sum + p.y, 0) / open.length
    return { x: meanX, y: meanY }
  }

  return { x: cx / (3 * crossSum), y: cy / (3 * crossSum) }
}

function chordLength(points: Point[]): number {
  const xs = points.map((point) => point.x)
  return Math.max(...xs) - Math.min(...xs)
}

function ensureCounterClockwise(points: Point[]): Point[] {
  return polygonArea(points) < 0 ? [...points].reverse() : points
}

function validateDomainFit(points: Point[]): void {
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)

  if (
    minX < DOMAIN_X_MIN + DOMAIN_MARGIN ||
    maxX > DOMAIN_X_MAX - DOMAIN_MARGIN ||
    minY < DOMAIN_Y_MIN + DOMAIN_MARGIN ||
    maxY > DOMAIN_Y_MAX - DOMAIN_MARGIN
  ) {
    throw new Error(
      'Normalized shape does not fit inside the prediction domain. Try a smaller drawing.',
    )
  }
}

export function rotatePoint(point: Point, angleDeg: number): Point {
  const theta = (angleDeg * Math.PI) / 180
  const cosine = Math.cos(theta)
  const sine = Math.sin(theta)
  return {
    x: point.x * cosine - point.y * sine,
    y: point.x * sine + point.y * cosine,
  }
}

/** Rotate body in physical coords (training uses R(-AoA) for horizontal freestream). */
export function rotateGeometry(points: Point[], aoaDeg: number): Point[] {
  const rotationDeg = -aoaDeg
  const rotated = points.map((point) => rotatePoint(point, rotationDeg))
  validateDomainFit(rotated)
  return rotated
}

export function normalizeGeometry(
  points: Point[],
  targetChord = TARGET_CHORD,
): Point[] {
  const open = openLoop(points)
  if (open.length < 3) {
    throw new Error('Shape must contain at least 3 points.')
  }

  const centroid = polygonCentroid(open)
  let centered = open.map((point) => ({
    x: point.x - centroid.x,
    y: point.y - centroid.y,
  }))

  const currentChord = chordLength(centered)
  if (currentChord <= 0) {
    throw new Error('Shape has zero chord length.')
  }

  const scale = targetChord / currentChord
  centered = centered.map((point) => ({
    x: point.x * scale,
    y: point.y * scale,
  }))

  const normalized = ensureCounterClockwise(centered)
  validateDomainFit(normalized)
  return normalized
}
