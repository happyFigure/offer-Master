import { apiRequest, toQueryString } from "./client";
import type { OfferIOCompany, OfferIOCompanyOpening, OfferIOJob, OfferIOPage } from "../types/jobs";


const OFFERIO_SOURCE_TIMEOUT_MS = 25_000;


export interface OfferIOCompanyFilters {
  job_type?: string;
  page?: number;
  page_size?: number;
  keyword?: string;
  industry?: string;
}

export interface OfferIOCompanyOpeningFilters {
  page?: number;
  page_size?: number;
  keyword?: string;
  industry?: string;
  batch?: string;
  target?: string;
  company_nature?: string;
}

export interface OfferIOJobFilters {
  job_type?: string;
  company?: string;
  keyword?: string;
  page?: number;
  page_size?: number;
}

export async function listOfferIOCompanies(filters: OfferIOCompanyFilters = {}): Promise<OfferIOPage<OfferIOCompany>> {
  const query = toQueryString({
    job_type: filters.job_type ?? "校招",
    page: filters.page ?? 1,
    page_size: filters.page_size ?? 20,
    keyword: filters.keyword,
    industry: filters.industry,
  });
  return apiRequest<OfferIOPage<OfferIOCompany>>(`/api/v1/jobs/offerio/companies${query}`, { timeoutMs: OFFERIO_SOURCE_TIMEOUT_MS });
}

export async function listOfferIOCompanyOpenings(filters: OfferIOCompanyOpeningFilters = {}): Promise<OfferIOPage<OfferIOCompanyOpening>> {
  const query = toQueryString({
    page: filters.page ?? 1,
    page_size: filters.page_size ?? 50,
    keyword: filters.keyword,
    industry: filters.industry,
    batch: filters.batch,
    target: filters.target,
    company_nature: filters.company_nature,
  });
  return apiRequest<OfferIOPage<OfferIOCompanyOpening>>(`/api/v1/jobs/offerio/company-openings${query}`, { timeoutMs: OFFERIO_SOURCE_TIMEOUT_MS });
}

export async function listOfferIOJobs(filters: OfferIOJobFilters = {}): Promise<OfferIOPage<OfferIOJob>> {
  const query = toQueryString({
    job_type: filters.job_type ?? "校招",
    company: filters.company,
    keyword: filters.keyword,
    page: filters.page ?? 1,
    page_size: filters.page_size ?? 50,
  });
  return apiRequest<OfferIOPage<OfferIOJob>>(`/api/v1/jobs/offerio/jobs${query}`, { timeoutMs: OFFERIO_SOURCE_TIMEOUT_MS });
}
