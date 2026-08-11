const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 8_000);
  const headers = new Headers(options.headers);

  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
      signal: options.signal ?? controller.signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("后端请求超时，请确认 FastAPI 服务和 Nginx 代理状态。", 408);
    }
    throw new ApiError("后端服务暂不可用，请确认 FastAPI 服务已启动。", 0, error);
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const details = await parseResponse(response);
    throw new ApiError(resolveApiMessage(details, response.status, response.statusText), response.status, details);
  }

  return parseResponse(response) as Promise<T>;
}

export function toQueryString(params: Record<string, string | number | undefined>): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      searchParams.set(key, String(value));
    }
  });

  const query = searchParams.toString();
  return query ? `?${query}` : "";
}

async function parseResponse(response: Response): Promise<unknown> {
  const contentType = response.headers.get("Content-Type") ?? "";

  if (response.status === 204) {
    return undefined;
  }

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

function resolveApiMessage(details: unknown, status: number, fallback: string): string {
  if (typeof details === "string" && details.trim()) {
    const trimmed = details.trim();
    if (trimmed.startsWith("<") || trimmed.toLowerCase().includes("<!doctype html")) {
      return `后端代理返回 HTTP ${status}，请确认 FastAPI 服务已在 127.0.0.1:8000 启动。`;
    }
    return trimmed.length > 180 ? `${trimmed.slice(0, 180)}...` : trimmed;
  }

  if (typeof details === "object" && details !== null && "detail" in details) {
    const detail = (details as { detail?: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
  }

  return fallback || `HTTP ${status} request failed`;
}
