import { NextRequest } from "next/server";
import { proxyJson } from "../../../../_proxy";

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ request_id: string }> }
) {
  const { request_id } = await ctx.params;
  return proxyJson(req, `/ask/runs/${encodeURIComponent(request_id)}/bundle`);
}
