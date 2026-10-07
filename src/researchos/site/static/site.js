// ResearchOS project site. Progressive enhancement only: the page is complete without it.
(function () {
  "use strict";

  var root = document.documentElement;
  root.classList.add("js");

  // Storage can be unavailable (private windows, file:// in some browsers).
  function load(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }
  function save(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* not persisted */ }
  }

  // Theme ------------------------------------------------------------------------------
  var THEMES = ["system", "light", "dark"];
  var LABELS = { system: "Theme: match system", light: "Theme: light", dark: "Theme: dark" };
  var themeButton = document.getElementById("theme-toggle");

  function applyTheme(theme) {
    if (theme === "system") { root.removeAttribute("data-theme"); }
    else { root.setAttribute("data-theme", theme); }
    themeButton.textContent = LABELS[theme];
  }
  var theme = load("researchos-theme");
  if (THEMES.indexOf(theme) === -1) { theme = "system"; }
  applyTheme(theme);
  themeButton.addEventListener("click", function () {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
    applyTheme(theme);
    save("researchos-theme", theme);
  });

  // Contents toggle on narrow screens ----------------------------------------------------
  var rail = document.getElementById("rail");
  var railToggle = rail.querySelector(".rail-toggle");
  function setRail(open) {
    rail.classList.toggle("open", open);
    railToggle.setAttribute("aria-expanded", String(open));
  }
  railToggle.addEventListener("click", function () {
    setRail(!rail.classList.contains("open"));
  });
  rail.querySelector("nav").addEventListener("click", function (event) {
    if (event.target.closest("a")) { setRail(false); }
  });

  // Search -------------------------------------------------------------------------------
  var input = document.getElementById("search");
  var panel = document.getElementById("search-results");
  var dataElement = document.getElementById("search-data");
  var entries = [];
  try { entries = JSON.parse(dataElement.textContent || "[]"); } catch (e) { entries = []; }
  entries.forEach(function (entry) {
    entry.haystack = (entry.title + " " + entry.text).toLowerCase();
  });

  var MAX_RESULTS = 12;
  var selected = -1;

  function terms(query) {
    return query.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 0; });
  }

  // Append `text` to `parent`, wrapping occurrences of any term in <mark>. Built from text
  // nodes so search data is never interpreted as HTML.
  function appendHighlighted(parent, text, words) {
    if (!words.length) { parent.appendChild(document.createTextNode(text)); return; }
    var escaped = words.map(function (w) { return w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); });
    var pattern = new RegExp("(" + escaped.join("|") + ")", "gi");
    var last = 0;
    text.replace(pattern, function (match, _group, offset) {
      parent.appendChild(document.createTextNode(text.slice(last, offset)));
      var mark = document.createElement("mark");
      mark.textContent = match;
      parent.appendChild(mark);
      last = offset + match.length;
      return match;
    });
    parent.appendChild(document.createTextNode(text.slice(last)));
  }

  function snippet(text, words) {
    var lower = text.toLowerCase();
    var at = words.length ? lower.indexOf(words[0]) : 0;
    var start = Math.max(0, at - 50);
    var end = Math.min(text.length, start + 160);
    return (start > 0 ? "…" : "") + text.slice(start, end) + (end < text.length ? "…" : "");
  }

  function render(query) {
    var words = terms(query);
    panel.textContent = "";
    selected = -1;
    if (!words.length) { panel.hidden = true; return; }

    var matches = entries.filter(function (entry) {
      return words.every(function (w) { return entry.haystack.indexOf(w) !== -1; });
    }).slice(0, MAX_RESULTS);

    if (!matches.length) {
      var empty = document.createElement("p");
      empty.className = "result-empty";
      empty.textContent = "Nothing matches “" + query.trim() + "”.";
      panel.appendChild(empty);
    }
    matches.forEach(function (entry, i) {
      var link = document.createElement("a");
      link.href = entry.href;
      link.id = "search-result-" + i;
      var kind = document.createElement("span");
      kind.className = "result-kind";
      kind.textContent = entry.kind;
      var title = document.createElement("span");
      title.className = "result-title";
      appendHighlighted(title, entry.title, words);
      var text = document.createElement("span");
      text.className = "result-snippet";
      appendHighlighted(text, snippet(entry.text, words), words);
      link.appendChild(kind);
      link.appendChild(title);
      link.appendChild(text);
      panel.appendChild(link);
    });
    panel.hidden = false;
  }

  function links() { return panel.querySelectorAll("a"); }

  function select(index) {
    var all = links();
    if (!all.length) { return; }
    selected = (index + all.length) % all.length;
    all.forEach(function (a, i) { a.setAttribute("aria-selected", String(i === selected)); });
    all[selected].scrollIntoView({ block: "nearest" });
  }

  function close() { panel.hidden = true; selected = -1; }

  input.addEventListener("input", function () { render(input.value); });
  input.addEventListener("keydown", function (event) {
    if (event.key === "ArrowDown") { event.preventDefault(); select(selected + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); select(selected - 1); }
    else if (event.key === "Enter") {
      var all = links();
      var target = all[selected >= 0 ? selected : 0];
      if (target) { event.preventDefault(); target.click(); }
    } else if (event.key === "Escape") { close(); input.blur(); }
  });
  panel.addEventListener("click", function (event) {
    if (event.target.closest("a")) { close(); setRail(false); }
  });
  document.addEventListener("click", function (event) {
    if (!event.target.closest(".search")) { close(); }
  });
  document.addEventListener("keydown", function (event) {
    var typing = /^(input|textarea|select)$/i.test(event.target.tagName) || event.target.isContentEditable;
    if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      if (window.matchMedia("(max-width: 820px)").matches) { setRail(true); }
      input.focus();
    }
  });

  // Highlight the section being read in the contents ---------------------------------------
  if ("IntersectionObserver" in window) {
    var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc a[href^='#']"));
    var byId = {};
    tocLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var targets = Object.keys(byId)
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);
    var current = null;
    var observer = new IntersectionObserver(function (records) {
      records.forEach(function (record) {
        if (!record.isIntersecting) { return; }
        if (current) { current.removeAttribute("aria-current"); }
        current = byId[record.target.id];
        current.setAttribute("aria-current", "true");
      });
    }, { rootMargin: "0px 0px -75% 0px" });
    targets.forEach(function (t) { observer.observe(t); });
  }
})();
