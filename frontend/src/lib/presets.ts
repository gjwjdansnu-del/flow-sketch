/** Preset body shapes in physical coordinates (x, y). */

export type PhysicalPoint = { x: number; y: number }

export type PresetId =
  | 'diamond'
  | 'wedge'
  | 'ellipse'
  | 'blunt_capsule'
  | 'biconvex'
  | 'thin_plate'

export type PresetOption = {
  id: PresetId
  label: string
}

export const PRESET_OPTIONS: PresetOption[] = [
  { id: 'diamond', label: 'Diamond airfoil' },
  { id: 'wedge', label: 'Wedge' },
  { id: 'ellipse', label: 'Ellipse' },
  { id: 'blunt_capsule', label: 'Blunt capsule' },
  { id: 'biconvex', label: 'Biconvex airfoil' },
  { id: 'thin_plate', label: 'Thin plate' },
]

function diamondAirfoil(): PhysicalPoint[] {
  return [
    { x: -0.5, y: 0 },
    { x: 0, y: -0.08 },
    { x: 0.5, y: 0 },
    { x: 0, y: 0.08 },
  ]
}

function wedge(): PhysicalPoint[] {
  return [
    { x: -0.5, y: -0.08 },
    { x: 0.5, y: 0 },
    { x: -0.5, y: 0.08 },
  ]
}

function ellipse(pointCount = 80): PhysicalPoint[] {
  const rx = 0.5
  const ry = 0.12
  const points: PhysicalPoint[] = []
  for (let i = 0; i < pointCount; i++) {
    const t = (i / pointCount) * Math.PI * 2
    points.push({
      x: rx * Math.cos(t),
      y: ry * Math.sin(t),
    })
  }
  return points
}

function bluntCapsule(arcSegments = 32, lineSegments = 8): PhysicalPoint[] {
  const radius = 0.12
  const halfBody = 0.5 - radius
  const points: PhysicalPoint[] = []

  // Right semicircle (bottom → top along the right cap)
  for (let i = 0; i < arcSegments; i++) {
    const angle = -Math.PI / 2 + (i / arcSegments) * Math.PI
    points.push({
      x: halfBody + radius * Math.cos(angle),
      y: radius * Math.sin(angle),
    })
  }

  // Top straight segment (right → left)
  for (let i = 0; i < lineSegments; i++) {
    const t = i / lineSegments
    points.push({ x: halfBody * (1 - 2 * t), y: radius })
  }

  // Left semicircle (top → bottom along the left cap)
  for (let i = 0; i < arcSegments; i++) {
    const angle = Math.PI / 2 + (i / arcSegments) * Math.PI
    points.push({
      x: -halfBody + radius * Math.cos(angle),
      y: radius * Math.sin(angle),
    })
  }

  // Bottom straight segment (left → right)
  for (let i = 0; i < lineSegments; i++) {
    const t = i / lineSegments
    points.push({ x: -halfBody + 2 * halfBody * t, y: -radius })
  }

  return points
}

function biconvexAirfoil(samples = 64): PhysicalPoint[] {
  const thickness = 0.12
  const points: PhysicalPoint[] = []

  for (let i = 0; i <= samples; i++) {
    const x = -0.5 + (i / samples) * 1.0
    const y = thickness * (1 - (2 * x) ** 2)
    points.push({ x, y })
  }

  for (let i = samples; i >= 0; i--) {
    const x = -0.5 + (i / samples) * 1.0
    const y = -thickness * (1 - (2 * x) ** 2)
    points.push({ x, y })
  }

  return points
}

function thinPlate(): PhysicalPoint[] {
  return [
    { x: -0.5, y: -0.015 },
    { x: 0.5, y: -0.015 },
    { x: 0.5, y: 0.015 },
    { x: -0.5, y: 0.015 },
  ]
}

const PRESET_BUILDERS: Record<PresetId, () => PhysicalPoint[]> = {
  diamond: diamondAirfoil,
  wedge,
  ellipse,
  blunt_capsule: bluntCapsule,
  biconvex: biconvexAirfoil,
  thin_plate: thinPlate,
}

export function getPresetPolygon(presetId: PresetId): PhysicalPoint[] {
  return PRESET_BUILDERS[presetId]()
}
