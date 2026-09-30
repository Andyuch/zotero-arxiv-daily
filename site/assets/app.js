const state = {
  data: { papers: [], journals: [], topics: [], days: [] },
  query: "",
  journal: "",
  date: "",
  topic: "All topics",
  sort: "recent",
};

const $ = (id) => document.getElementById(id);

function safeUrl(value) {
  if (!value) return "";
  try {
    const u = new URL(value, window.location.href);
    return ["http:", "https:"].includes(u.protocol) ? u.href : "";
  } catch {
    return "";
  }
}

function relevanceStars(percentile = 0) {
  const p = Number(percentile) || 0;
  let n = 2;
  if (p >= 95) n = 5;
  else if (p >= 85) n = 4.5;
  else if (p >= 70) n = 4;
  else if (p >= 50) n = 3.5;
  else if (p >= 30) n = 3;
  else if (p >= 15) n = 2.5;

  const full = Math.floor(n);
  const half = n % 1 ? "½" : "";
  return "★".repeat(full) + half;
}

function relevanceLabel(paper) {
  const percentile = Number(paper.relevance_percentile || 0);
  const top = Math.max(1, Math.round(100 - percentile));
  const score = Number(paper.score || 0).toFixed(2);
  return `Score ${score} · Top ${top}% of that day's candidate pool`;
}

function textOr(value, fallback = "—") {
  if (Array.isArray(value)) return value.length ? value.join(", ") : fallback;
  return value ? String(value) : fallback;
}

function setOptions(select, values, allLabel) {
  select.innerHTML = "";
  const first = document.createElement("option");
  first.value = "";
  first.textContent = allLabel;
  select.appendChild(first);
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

function buildTopics(topics) {
  const row = $("topic-row");
  row.innerHTML = "";
  ["All topics", ...topics].forEach((topic) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "topic-button" + (state.topic === topic ? " active" : "");
    button.textContent = topic;
    button.addEventListener("click", () => {
      state.topic = topic;
      buildTopics(state.data.topics || []);
      render();
    });
    row.appendChild(button);
  });
}

function matchQuery(paper) {
  if (!state.query.trim()) return true;
  const haystack = [
    paper.title,
    ...(paper.authors || []),
    paper.journal,
    paper.source,
    paper.abstract,
    paper.tldr,
    ...(paper.topics || []),
    ...(paper.affiliations || []),
    paper.doi,
    paper.arxiv_id,
  ].filter(Boolean).join(" ").toLowerCase();

  return state.query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => haystack.includes(term));
}

function filteredPapers() {
  const papers = [...(state.data.papers || [])].filter((paper) => {
    if (!matchQuery(paper)) return false;
    if (state.journal && paper.journal !== state.journal) return false;
    if (state.date && !(paper.seen_dates || [paper.seen_date]).includes(state.date)) return false;
    if (state.topic !== "All topics" && !(paper.topics || []).includes(state.topic)) return false;
    return true;
  });

  papers.sort((a, b) => {
    if (state.sort === "relevance") {
      return Number(b.relevance_percentile || 0) - Number(a.relevance_percentile || 0);
    }
    if (state.sort === "score") {
      return Number(b.score || 0) - Number(a.score || 0);
    }
    if (state.sort === "journal") {
      return String(a.journal || "").localeCompare(String(b.journal || ""));
    }
    return String(b.last_seen || b.seen_date || "").localeCompare(String(a.last_seen || a.seen_date || ""));
  });

  return papers;
}

function actionLink(label, href, primary = false) {
  const url = safeUrl(href);
  if (!url) return null;
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = label;
  if (primary) a.className = "primary";
  return a;
}

function renderCard(paper) {
  const tpl = $("paper-template");
  const node = tpl.content.cloneNode(true);

  node.querySelector(".journal-pill").textContent = paper.journal || paper.source || "Unknown source";
  node.querySelector(".date-pill").textContent = paper.published_at || paper.last_seen || paper.seen_date || "";
  node.querySelector(".paper-title").textContent = paper.title || "Untitled";
  node.querySelector(".paper-authors").textContent = textOr(paper.authors, "Unknown authors");
  node.querySelector(".stars").textContent = relevanceStars(paper.relevance_percentile);
  node.querySelector(".relevance-text").textContent = relevanceLabel(paper);
  const recommendation = (state.date && paper.recommendations_by_date?.[state.date]) || paper;
  if (["repeat_highlight", "revisit"].includes(recommendation.recommendation_status)) {
    const badge = document.createElement("span");
    badge.className = "topic-chip";
    const label = recommendation.recommendation_status === "repeat_highlight" ? "Repeat highlight" : "Revisit";
    badge.textContent = `${label} · previously recommended ${recommendation.previous_recommended_at || "earlier"}`;
    badge.title = recommendation.recommendation_status === "repeat_highlight"
      ? "Filling a shortfall of candidates outside the recent-recommendation cooldown"
      : "Recommended again after the cooldown";
    node.querySelector(".topic-list").appendChild(badge);
  }
  node.querySelector(".paper-tldr").textContent = paper.tldr || paper.abstract || "No summary available.";

  const topics = node.querySelector(".topic-list");
  (paper.topics || []).forEach((topic) => {
    const span = document.createElement("span");
    span.className = "topic-chip";
    span.textContent = topic;
    topics.appendChild(span);
  });

  node.querySelector(".paper-abstract").textContent = paper.abstract || "Abstract unavailable from source metadata.";
  node.querySelector(".paper-source").textContent = textOr(paper.source);
  node.querySelector(".paper-id").textContent = paper.doi || paper.arxiv_id || "—";
  node.querySelector(".paper-affiliations").textContent = textOr(paper.affiliations);

  const components = paper.score_components || {};
  const componentText = Object.entries(components)
    .map(([k, v]) => `${k.replaceAll("_", " ")} ${Number(v).toFixed(3)}`)
    .join(" · ");
  node.querySelector(".paper-components").textContent = componentText || "—";

  const actions = node.querySelector(".paper-actions");
  [
    actionLink("Article", paper.article_url, true),
    actionLink("PDF", paper.pdf_url),
    actionLink("Code", paper.code_url),
  ].filter(Boolean).forEach((link) => actions.appendChild(link));

  return node;
}

function render() {
  const list = $("paper-list");
  const papers = filteredPapers();
  list.innerHTML = "";
  papers.forEach((paper) => list.appendChild(renderCard(paper)));
  $("result-count").textContent = papers.length;
  $("empty-state").hidden = papers.length !== 0;
}

function updateStats() {
  const papers = state.data.papers || [];
  const latestDay = (state.data.days || [])[0]?.date || "";
  const todayCount = (state.data.days || [])[0]?.count || 0;
  const journalCount = new Set(papers.map((p) => p.journal || p.source).filter(Boolean)).size;
  const topCount = papers.filter((p) => Number(p.relevance_percentile || 0) >= 90).length;

  $("stat-total").textContent = papers.length;
  $("stat-today").textContent = todayCount;
  $("stat-journals").textContent = journalCount;
  $("stat-top").textContent = topCount;

  if (state.data.generated_at) {
    const dt = new Date(state.data.generated_at);
    $("updated-at").textContent = `Archive updated ${dt.toLocaleString()}`;
  } else if (latestDay) {
    $("updated-at").textContent = `Latest archive day: ${latestDay}`;
  }
}

function wireControls() {
  $("search-input").addEventListener("input", (e) => {
    state.query = e.target.value;
    render();
  });
  $("journal-filter").addEventListener("change", (e) => {
    state.journal = e.target.value;
    render();
  });
  $("date-filter").addEventListener("change", (e) => {
    state.date = e.target.value;
    render();
  });
  $("sort-filter").addEventListener("change", (e) => {
    state.sort = e.target.value;
    render();
  });
}

async function boot() {
  $("year-label").textContent = new Date().getFullYear();
  wireControls();

  try {
    const response = await fetch("data/index.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();

    setOptions($("journal-filter"), state.data.journals || [], "All journals");
    setOptions(
      $("date-filter"),
      (state.data.days || []).map((d) => d.date),
      "All dates"
    );
    buildTopics(state.data.topics || []);
    updateStats();
    render();
  } catch (error) {
    console.error("Failed to load archive data:", error);
    $("empty-state").hidden = false;
    $("empty-state").querySelector("h2").textContent = "Archive data is not available yet.";
    $("empty-state").querySelector("p").textContent =
      "The site shell is deployed. The first recommendation run will populate the archive.";
  }
}

boot();
