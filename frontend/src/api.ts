import { type FieldKey, type PredictResponse } from './lib/constants'
import { predictFlowInBrowser } from './lib/onnxInference'

export type PredictRequest = {
  solid_mask: number[][]
  mach: number
  /** Ignored; AoA is encoded in the rotated solid_mask. */
  aoa?: number
}

export async function predictFlow(
  payload: PredictRequest,
): Promise<PredictResponse> {
  return predictFlowInBrowser(payload.solid_mask, payload.mach)
}

export function pickField(
  response: PredictResponse,
  field: FieldKey,
): number[][] {
  return response[field]
}
