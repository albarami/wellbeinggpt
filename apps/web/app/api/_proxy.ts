import { NextRequest, NextResponse } from "next/server";

function requireApiBaseUrl(): string {
  const base = process.env.API_BASE_URL;
  if (!base) {
    throw new Error("API_BASE_URL is not set (server env)");
  }
  return base.replace(/\/+$/, "");
}

export async function proxyJson(
  req: NextRequest,
  backendPath: string,
  opts?: { method?: string; body?: unknown }
): Promise<NextResponse> {
  const base = requireApiBaseUrl();
  const method = opts?.method ?? req.method ?? "GET";

  const url = new URL(base + backendPath);
  const incoming = new URL(req.url);
  incoming.searchParams.forEach((v, k) => url.searchParams.append(k, v));

  const init: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: opts?.body !== undefined ? JSON.stringify(opts.body) : undefined,
    cache: "no-store",
  };

  const resp = await fetch(url.toString(), init);
  const text = await resp.text();

  return new NextResponse(text, {
    status: resp.status,
    headers: {
      "Content-Type": resp.headers.get("content-type") ?? "application/json",
    },
  });
}

