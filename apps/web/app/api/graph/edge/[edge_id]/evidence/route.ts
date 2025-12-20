import { NextRequest } from "next/server";
import { proxyJson } from "../../../../_proxy";

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ edge_id: string }> }
) {
  const { edge_id } = await ctx.params;
  return proxyJson(req, `/graph/edge/${encodeURIComponent(edge_id)}/evidence`);
}
