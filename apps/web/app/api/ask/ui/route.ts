import { NextRequest } from "next/server";
import { proxyJson } from "../../_proxy";

export async function POST(req: NextRequest) {
  const body = await req.json();
  return proxyJson(req, "/ask/ui", { method: "POST", body });
}

