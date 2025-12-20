import { NextRequest } from "next/server";
import { proxyJson } from "../../_proxy";

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ chunk_id: string }> }
) {
  const { chunk_id } = await ctx.params;
  return proxyJson(req, `/chunk/${encodeURIComponent(chunk_id)}`);
}
