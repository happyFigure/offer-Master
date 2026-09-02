import {
  importJobLeadsFromUrl,
  listDomainHealth,
  listDomainHealthByDomain,
  pollUrlImportRun,
  submitVisiblePageContent,
} from "./urlImport";
import type { ImportUrlAcceptedResponse, ImportUrlRun, UrlImportInput, VisiblePageContentInput } from "../types/jobs";

const input: UrlImportInput = {
  url: "https://career.example.com/jobs/java?utm_source=xhs",
  source_hint: "official_career_site",
  trust_level: "medium_high",
  force_refresh: false,
};

const accepted: Promise<ImportUrlAcceptedResponse> = importJobLeadsFromUrl(input);
const run: Promise<ImportUrlRun> = pollUrlImportRun("run-id");
const visibleInput: VisiblePageContentInput = { visible_text: "visible page text", title: "XHS note" };
const resumed: Promise<ImportUrlRun> = submitVisiblePageContent("run-id", visibleInput);
const allHealth = listDomainHealth();
const domainHealth = listDomainHealthByDomain("career.example.com");

void accepted;
void run;
void resumed;
void allHealth;
void domainHealth;
