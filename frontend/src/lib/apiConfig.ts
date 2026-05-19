/** Resolve backend API base URL for fetch calls. */

const configuredBase = (import.meta.env.VITE_API_BASE as string | undefined)?.trim() ?? ''

/** Trim trailing slash; empty string means use Vite dev proxy (relative URLs). */
export function resolveApiBase(): string {
  return configuredBase.replace(/\/$/, '')
}

export const API_BASE = resolveApiBase()

/** In production builds, VITE_API_BASE must point at the deployed API (e.g. Render). */
export function isApiBaseConfigured(): boolean {
  if (import.meta.env.DEV) {
    return true
  }
  return API_BASE.length > 0
}

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  if (!isApiBaseConfigured()) {
    throw new Error(
      'API backend URL is not configured. Set GitHub Actions variable VITE_API_BASE to your Render API URL and redeploy Pages.',
    )
  }
  return `${API_BASE}${normalizedPath}`
}

export function formatFetchError(error: unknown, endpoint: string): string {
  if (error instanceof Error && error.message.includes('API backend URL is not configured')) {
    return error.message
  }
  const target = isApiBaseConfigured() ? apiUrl(endpoint) : '(API URL not set)'
  if (error instanceof TypeError) {
    return (
      `Cannot reach the API at ${target}. ` +
      'Deploy the Render backend first, then open /health in your browser. "Failed to fetch" usually means the API is down or CORS is misconfigured.'
    )
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Request failed.'
}
