import { apiRequest, toQueryString } from "./client";
import type { ArticleCandidate, RecruitingSignal } from "../types/jobs";


interface ArticleCandidateListResponse {
  items: ArticleCandidate[];
}

interface RecruitingSignalListResponse {
  items: RecruitingSignal[];
}

export async function listArticleCandidates(filters: { source_id?: string; status?: string; limit?: number } = {}): Promise<ArticleCandidate[]> {
  const response = await apiRequest<ArticleCandidateListResponse>(`/api/v1/article-candidates${toQueryString({ ...filters, limit: filters.limit ?? 80 })}`);
  return response.items;
}

export async function listRecruitingSignals(filters: { source_id?: string; status?: string; company?: string; graduation_year?: string; limit?: number } = {}): Promise<RecruitingSignal[]> {
  const response = await apiRequest<RecruitingSignalListResponse>(`/api/v1/recruiting-signals${toQueryString({ ...filters, limit: filters.limit ?? 80 })}`);
  return response.items;
}
