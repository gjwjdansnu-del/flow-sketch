import { API_BASE, type FieldKey, type PredictResponse } from './lib/constants'

export type PredictRequest = {
  solid_mask: number[][]
  mach: number
  /** Ignored by rotated2 backend; AoA is encoded in the rotated solid_mask. */
  aoa?: number
}

export async function predictFlow(
  payload: PredictRequest,
): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Predict failed (${response.status})`)
  }

  return (await response.json()) as PredictResponse
}

export async function fetchHealth(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`)
  }
  return (await response.json()) as Record<string, unknown>
}

export function pickField(
  response: PredictResponse,
  field: FieldKey,
): number[][] {
  return response[field]
}
