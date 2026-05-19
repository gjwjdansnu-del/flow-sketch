import {
  API_BASE,
  apiUrl,
  formatFetchError,
  isApiBaseConfigured,
} from './lib/apiConfig'
import { type FieldKey, type PredictResponse } from './lib/constants'

export type PredictRequest = {
  solid_mask: number[][]
  mach: number
  /** Ignored by rotated2 backend; AoA is encoded in the rotated solid_mask. */
  aoa?: number
}

export { API_BASE, isApiBaseConfigured }

export async function predictFlow(
  payload: PredictRequest,
): Promise<PredictResponse> {
  let url: string
  try {
    url = apiUrl('/predict')
  } catch (configError) {
    throw configError
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || `Predict failed (${response.status})`)
    }

    return (await response.json()) as PredictResponse
  } catch (error) {
    throw new Error(formatFetchError(error, '/predict'))
  }
}

export async function fetchHealth(): Promise<Record<string, unknown>> {
  if (!isApiBaseConfigured()) {
    throw new Error(
      'API backend URL is not configured. Set VITE_API_BASE and redeploy GitHub Pages.',
    )
  }

  try {
    const response = await fetch(apiUrl('/health'))
    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`)
    }
    return (await response.json()) as Record<string, unknown>
  } catch (error) {
    throw new Error(formatFetchError(error, '/health'))
  }
}

export function pickField(
  response: PredictResponse,
  field: FieldKey,
): number[][] {
  return response[field]
}
