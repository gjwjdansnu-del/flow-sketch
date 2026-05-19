export const DISPLAY_WIDTH = 512
export const DISPLAY_HEIGHT = 256

export const GRID_WIDTH = 256
export const GRID_HEIGHT = 128

export const MASK_ROWS = 128
export const MASK_COLS = 256

export const X_MIN = -1
export const X_MAX = 3
export const Y_MIN = -1
export const Y_MAX = 1

export const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export type FieldKey =
  | 'mach'
  | 'pressure'
  | 'density'
  | 'temperature'
  | 'shock_indicator'

export const FIELD_OPTIONS: { key: FieldKey; label: string }[] = [
  { key: 'mach', label: 'Mach' },
  { key: 'pressure', label: 'Pressure' },
  { key: 'density', label: 'Density' },
  { key: 'temperature', label: 'Temperature' },
  { key: 'shock_indicator', label: 'Shock' },
]

export type PredictResponse = Record<FieldKey, number[][]>
