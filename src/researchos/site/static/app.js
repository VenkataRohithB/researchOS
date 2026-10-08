// ResearchOS learning site.
//
// All content comes from the JSON embedded in the page. Fields whose names end in "Html" are
// markup rendered and escaped by the site builder; everything else is plain text and is only
// ever inserted with textContent.
(function () {
  "use strict";

  var data = JSON.parse(document.getElementById("data").textContent);
  var view = document.getElementById("view");
  var body = document.body;
  var root = document.documentElement;
  var LEVELS = data.levels.map(function (l) { return l.id; });
  var LABELS = {};
  data.levels.forEach(function (l) { LABELS[l.id] = l.label; });

  // Storage can be unavailable (private windows, file:// in some browsers).
  function load(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } }
  function save(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* not kept */ } }
  var KEY = "researchos:" + data.projectId + ":";

  // DOM helpers ----------------------------------------------------------------------------

  function el(tag, props, children) {
    var node = document.createElement(tag);
    Object.keys(props || {}).forEach(function (key) {
      var value = props[key];
      if (value === null || value === undefined || value === false) { return; }
      if (key === "text") { node.textContent = value; }
      else if (key === "html") { node.innerHTML = value; } // only for builder-rendered *Html fields
      else if (key === "class") { node.className = value; }
      else { node.setAttribute(key, value === true ? "" : value); }
    });
    (children || []).forEach(function (child) {
      if (child === null || child === undefined || child === false) { return; }
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  // replaceChildren() would turn a null into the text "null"; skip empty pieces instead.
  function fill(parent) {
    var nodes = Array.prototype.slice.call(arguments, 1).filter(function (n) { return n !== null && n !== undefined && n !== false; });
    parent.replaceChildren.apply(parent, nodes);
  }

  function plain(html) {
    var holder = document.createElement("div");
    holder.innerHTML = html || ""; // builder-rendered markup
    return holder.textContent || "";
  }

  function concept(id) { return data.concepts[id]; }
  function link(id) { return "#/concept/" + encodeURIComponent(id); }

  // Depth -----------------------------------------------------------------------------------

  var depth = LEVELS.indexOf(load(KEY + "depth")) >= 0 ? load(KEY + "depth") : "beginner";

  function setDepth(level) {
    depth = level;
    body.setAttribute("data-depth", level);
    save(KEY + "depth", level);
  }
  setDepth(depth);

  // The level to show for a concept: the chosen depth, or the nearest one that exists.
  function shownLevel(c) {
    if (c.levels[depth]) { return depth; }
    var at = LEVELS.indexOf(depth);
    for (var d = 1; d < LEVELS.length; d++) {
      var shallower = LEVELS[at - d], deeper = LEVELS[at + d];
      if (shallower && c.levels[shallower]) { return shallower; }
      if (deeper && c.levels[deeper]) { return deeper; }
    }
    return null;
  }

  function depthControl(c, onChange) {
    var available = c ? c.levels : null;
    var scale = el("div", { class: "depth-scale", role: "group", "aria-label": "Depth" });
    LEVELS.forEach(function (level, i) {
      var button = el("button", {
        type: "button",
        "data-level": level,
        "aria-pressed": String(level === depth),
        disabled: available && !available[level],
        class: i <= LEVELS.indexOf(depth) ? "reached" : null,
        text: LABELS[level]
      });
      button.addEventListener("click", function () { setDepth(level); onChange(); });
      scale.appendChild(button);
    });
    return el("div", { class: "depth" }, [
      scale,
      el("p", { class: "depth-hint", text: "Choose how deep to go. Your choice applies to every concept." })
    ]);
  }

  // Evidence drawer ---------------------------------------------------------------------------

  var drawer = document.getElementById("evidence");
  var openCite = null;

  function decorate(container) {
    container.querySelectorAll(".cite[data-claim]").forEach(function (button) {
      var claim = data.claims[button.getAttribute("data-claim")];
      if (claim) { button.setAttribute("data-status", claim.status); button.setAttribute("aria-expanded", "false"); }
    });
  }

  function showEvidence(claimId, trigger) {
    var claim = data.claims[claimId];
    if (!claim) { return; }
    if (openCite) { openCite.setAttribute("aria-expanded", "false"); }
    openCite = trigger;
    if (trigger) { trigger.setAttribute("aria-expanded", "true"); }
    var close = el("button", { type: "button", class: "evidence-close", "aria-label": "Close evidence", text: "×" });
    close.addEventListener("click", hideEvidence);
    var kind = claim.kind === "fact" ? null : (claim.kind === "inference" ? "An inference drawn from the sources" : "An interpretation of the sources");
    var items = claim.evidence.map(function (item) {
      var source = data.sources[item.source - 1];
      return el("li", { "data-stance": item.stance }, [
        el("blockquote", { text: item.quote }),
        el("p", { class: "evidence-source" }, [
          (item.stance === "supports" ? "Supports" : item.stance === "contradicts" ? "Contradicts" : "Qualifies") + ", from ",
          el("a", { href: "#/sources/" + item.source, text: "source " + item.source }),
          source ? " — " + source.domain + " (" + source.tier + ")" : ""
        ])
      ]);
    });
    fill(drawer,
      el("div", { class: "evidence-head" }, [
        el("span", { class: "evidence-status s-" + claim.status, text: claim.statusLabel }),
        close
      ]),
      el("p", { class: "evidence-claim", html: claim.textHtml }),
      kind ? el("p", { class: "evidence-kind", text: kind }) : null,
      items.length ? el("ol", {}, items) : el("p", { class: "evidence-kind", text: "No quoted evidence." }),
      claim.note ? el("p", { class: "evidence-note", text: claim.note }) : null
    );
    drawer.hidden = false;
    close.focus({ preventScroll: true });
  }

  function hideEvidence() {
    drawer.hidden = true;
    if (openCite) { openCite.setAttribute("aria-expanded", "false"); openCite.focus({ preventScroll: true }); openCite = null; }
  }

  document.addEventListener("click", function (event) {
    var cite = event.target.closest(".cite[data-claim]");
    if (cite) { event.preventDefault(); showEvidence(cite.getAttribute("data-claim"), cite); return; }
    if (!drawer.hidden && !event.target.closest("#evidence")) { hideEvidence(); }
  });

  // Views -------------------------------------------------------------------------------------

  function claimMeter() {
    var counts = data.statusCounts;
    var total = Object.keys(counts).reduce(function (sum, k) { return sum + counts[k]; }, 0) || 1;
    function part(cls, n) { var s = el("span", { class: cls }); s.style.width = (100 * n / total) + "%"; return s; }
    return el("div", {}, [
      el("div", { class: "meter", role: "img", "aria-label": counts.verified + " verified, " + counts.single_source + " single-source, " + counts.disputed + " disputed claims" }, [
        part("m-verified", counts.verified), part("m-single", counts.single_source), part("m-disputed", counts.disputed)
      ]),
      el("p", { class: "meter-legend", text: counts.verified + " verified, " + counts.single_source + " single-source, " + counts.disputed + " disputed" })
    ]);
  }

  function overview() {
    var path = data.path;
    var resume = load(KEY + "last");
    var start = resume && concept(resume) ? resume : path[0];
    var claimCount = Object.keys(data.claims).length;
    var actions = el("div", { class: "hero-actions" }, [
      start ? el("a", { class: "button primary", href: link(start), text: resume && concept(resume) ? "Continue with " + concept(resume).title : "Start learning" }) : null,
      el("a", { class: "button", href: "#/map", text: "Explore the concept map" })
    ]);
    var steps = path.map(function (id) {
      var c = concept(id);
      return el("li", {}, [
        el("a", { href: link(id), text: c.title }),
        c.children.length ? el("span", { class: "depth-tag", text: c.children.length + " sub-concept" + (c.children.length === 1 ? "" : "s") }) : null,
        c.summaryHtml ? el("p", { html: c.summaryHtml }) : null
      ]);
    });
    var facts = el("aside", { class: "facts" }, [
      el("h2", { text: "What this is built on" }),
      el("dl", {}, [
        el("dt", { text: String(path.length) }), el("dd", { text: "concepts to learn" }),
        el("dt", { text: String(claimCount) }), el("dd", { text: "claims, each quoted from a source" }),
        el("dt", { text: String(data.sources.length) }), el("dd", { text: "sources read" })
      ]),
      claimCount ? claimMeter() : null,
      el("p", { class: "meter-legend" }, ["Researched on " + data.researchedOn + ". ", el("a", { href: "#/sources", text: "See sources" })])
    ]);
    fill(view,
      el("section", { class: "hero" }, [
        el("h1", { text: data.topic }),
        data.goal ? el("p", { class: "hero-goal", text: data.goal }) : null,
        el("p", { class: "hero-meta", text: data.level ? "Written for: " + data.level + "." : "" }),
        actions
      ]),
      data.stopped ? el("div", { class: "notice", role: "note" }, [
        el("p", {}, [el("strong", { text: "This research is unfinished. " }), data.stopped + "."]),
        el("p", { text: "Continue it with: researchos resume " + data.projectId })
      ]) : null,
      el("div", { class: "overview" }, [
        el("section", {}, [
          el("h2", { class: "section-title", text: "Your learning path" }),
          el("p", { class: "section-lede", text: path.length ? "Concepts in the order to learn them: what each one needs comes first." : "No concepts have been researched yet." }),
          el("ol", { class: "path" }, steps)
        ]),
        facts
      ]),
      data.summaryHtml ? el("section", { class: "prose" }, [
        el("h2", { class: "section-title", text: "The research in brief" }),
        el("p", { class: "section-lede", text: "The agent's synthesis. Open any concept for explanations grounded in quoted evidence." }),
        el("div", { html: data.summaryHtml })
      ]) : null,
      footer()
    );
  }

  function crumbs(c) {
    var chain = [];
    for (var at = c; at; at = at.parent ? concept(at.parent) : null) { chain.unshift(at); }
    var parts = [el("a", { href: "#/", text: data.topic })];
    chain.forEach(function (node, i) {
      parts.push(el("span", { class: "sep", "aria-hidden": "true", text: "/" }));
      parts.push(i === chain.length - 1 ? el("span", { "aria-current": "page", text: node.title }) : el("a", { href: link(node.id), text: node.title }));
    });
    return el("nav", { class: "crumbs", "aria-label": "Where you are" }, parts);
  }

  function conceptView(id) {
    var c = concept(id);
    if (!c) { return notFound(); }
    save(KEY + "last", id);
    var level = shownLevel(c);
    var explanation = level
      ? el("div", { class: "explanation prose", html: c.levels[level] })
      : el("p", { class: "explanation explanation-missing", text: c.researched ? "Explanations for this concept have not been written yet." : "This concept was mentioned but not researched yet. Ask for it with: researchos improve " + data.projectId + " \"go deeper on " + c.title + "\"" });
    var note = level && level !== depth ? el("p", { class: "depth-hint", text: LABELS[depth] + " is not available for this concept; showing " + LABELS[level] + "." }) : null;

    var at = data.path.indexOf(id);
    var prev = at > 0 ? concept(data.path[at - 1]) : null;
    var next = at >= 0 && at < data.path.length - 1 ? concept(data.path[at + 1]) : null;

    function list(ids, cls) {
      return el("ul", { class: cls }, ids.map(function (cid) {
        var k = concept(cid);
        return el("li", {}, [el("a", { href: link(cid) }, [k.title, cls === "deeper" && k.summaryHtml ? el("span", { html: k.summaryHtml }) : null])]);
      }));
    }

    var claims = c.claims.map(function (cid) {
      var claim = data.claims[cid];
      return claim ? el("li", {}, [
        el("button", { type: "button", class: "cite", "data-claim": cid, "data-status": claim.status, "aria-label": "Evidence for claim " + claim.number, text: String(claim.number) }),
        el("span", { html: claim.textHtml })
      ]) : null;
    });

    var side = el("aside", { class: "side" }, [
      c.prerequisites.length ? el("section", {}, [el("h2", { text: "Learn first" }), el("div", { class: "chips" }, c.prerequisites.map(function (p) { return el("a", { href: link(p), text: concept(p).title }); }))]) : null,
      c.children.length ? el("section", {}, [el("h2", { text: "Go deeper" }), list(c.children, "deeper")]) : null,
      c.related.length ? el("section", {}, [el("h2", { text: "Related" }), el("div", { class: "chips" }, c.related.map(function (r) { return el("a", { href: link(r), text: concept(r).title }); }))]) : null,
      el("section", {}, [el("h2", { text: "Evidence about " + c.title }), claims.length ? el("ul", { class: "claim-list" }, claims) : el("p", { class: "empty", text: "No claims recorded yet." })])
    ]);

    fill(view,
      crumbs(c),
      el("div", { class: "concept" }, [
        el("article", {}, [
          el("h1", { text: c.title }),
          c.summaryHtml ? el("p", { class: "concept-summary", html: c.summaryHtml }) : null,
          depthControl(c, function () { conceptView(id); }),
          note,
          explanation,
          el("nav", { class: "go", "aria-label": "Learning path" }, [
            prev ? el("a", { href: link(prev.id) }, [el("small", { text: "Previous" }), prev.title]) : el("span"),
            next ? el("a", { href: link(next.id) }, [el("small", { text: "Next" }), next.title]) : el("a", { href: "#/revise" }, [el("small", { text: "Finished the path" }), "Revise what you learned"])
          ])
        ]),
        side
      ])
    );
    decorate(view);
  }

  // Concept map -----------------------------------------------------------------------------

  var SVG = "http://www.w3.org/2000/svg";
  function svg(tag, attrs) {
    var node = document.createElementNS(SVG, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function mapView() {
    var m = data.map;
    var nodes = {};
    m.nodes.forEach(function (n) { nodes[n.id] = n; });
    var box = m.viewBox.slice();
    var home = m.viewBox.slice();
    var canvas = svg("svg", { class: "map", viewBox: box.join(" "), role: "img", "aria-label": "Concept map of " + data.topic });
    var edgeLayer = svg("g"), nodeLayer = svg("g");
    canvas.appendChild(edgeLayer); canvas.appendChild(nodeLayer);
    var neighbours = {};
    function connect(a, b) { (neighbours[a] = neighbours[a] || []).push(b); (neighbours[b] = neighbours[b] || []).push(a); }

    m.edges.forEach(function (e) {
      var a = nodes[e.source], b = nodes[e.target];
      if (!a || !b) { return; }
      connect(a.id, b.id);
      var path = e.kind === "prerequisite"
        ? "M" + a.x + " " + a.y + " Q " + ((a.x + b.x) / 2 * 0.6) + " " + ((a.y + b.y) / 2 * 0.6) + " " + b.x + " " + b.y
        : "M" + a.x + " " + a.y + " L " + b.x + " " + b.y;
      var line = svg("path", { d: path, class: "edge " + e.kind, "data-a": a.id, "data-b": b.id });
      edgeLayer.appendChild(line);
    });

    var tip = el("div", { class: "map-tip", hidden: true });
    m.nodes.forEach(function (n) {
      var isRoot = n.id === m.root;
      var group = svg("g", { class: "node" + (isRoot ? " root" : "") + (n.researched ? "" : " stub"), transform: "translate(" + n.x + " " + n.y + ")", tabindex: "0", role: "link", "aria-label": n.title, "data-id": n.id });
      group.appendChild(svg("circle", { r: n.r }));
      // The topic is the page heading; its node is labelled on hover so it never crowds the ring.
      if (!isRoot) {
        var label = svg("text", { "text-anchor": n.x >= 0 ? "start" : "end", x: n.x >= 0 ? n.r + 8 : -n.r - 8, y: 5 });
        label.textContent = n.title.length > 34 ? n.title.slice(0, 32) + "…" : n.title;
        group.appendChild(label);
      }
      function go() { location.hash = isRoot ? "#/" : link(n.id); }
      group.addEventListener("click", function (ev) { if (!moved) { go(); } ev.stopPropagation(); });
      group.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); go(); } });
      group.addEventListener("pointerenter", function () { focusNode(n.id, true); });
      group.addEventListener("pointerleave", function () { focusNode(n.id, false); });
      group.addEventListener("focus", function () { focusNode(n.id, true); });
      group.addEventListener("blur", function () { focusNode(n.id, false); });
      nodeLayer.appendChild(group);
    });

    function focusNode(id, on) {
      canvas.classList.toggle("focusing", on);
      var near = {}; near[id] = true;
      (neighbours[id] || []).forEach(function (other) { near[other] = true; });
      nodeLayer.querySelectorAll(".node").forEach(function (g) { g.classList.toggle("near", on && !!near[g.getAttribute("data-id")]); });
      edgeLayer.querySelectorAll(".edge").forEach(function (p) { p.classList.toggle("near", on && (p.getAttribute("data-a") === id || p.getAttribute("data-b") === id)); });
      var c = id === m.root ? { title: data.topic, summaryHtml: "", researched: true } : concept(id);
      if (on && c) {
        fill(tip, el("strong", { text: c.title }), c.summaryHtml ? el("span", { html: c.summaryHtml }) : el("span", { text: c.researched ? "" : "Not researched yet" }));
        tip.hidden = false;
        var rect = canvas.getBoundingClientRect(), n = nodes[id];
        tip.style.left = Math.min(rect.width - 300, Math.max(8, (n.x - box[0]) / box[2] * rect.width + 16)) + "px";
        tip.style.top = Math.max(8, (n.y - box[1]) / box[3] * rect.height + 16) + "px";
      } else { tip.hidden = true; }
    }

    // Pan and zoom by changing the viewBox.
    var moved = false, dragging = null;
    function apply() { canvas.setAttribute("viewBox", box.join(" ")); }
    function zoom(factor, cx, cy) {
      var w = box[2] * factor, h = box[3] * factor;
      if (w < home[2] / 6 || w > home[2] * 3) { return; }
      box = [cx - (cx - box[0]) * factor, cy - (cy - box[1]) * factor, w, h];
      apply();
    }
    function toModel(ev) {
      var rect = canvas.getBoundingClientRect();
      return [box[0] + (ev.clientX - rect.left) / rect.width * box[2], box[1] + (ev.clientY - rect.top) / rect.height * box[3]];
    }
    canvas.addEventListener("wheel", function (ev) { ev.preventDefault(); var p = toModel(ev); zoom(ev.deltaY > 0 ? 1.12 : 1 / 1.12, p[0], p[1]); }, { passive: false });
    canvas.addEventListener("pointerdown", function (ev) { dragging = { x: ev.clientX, y: ev.clientY, box: box.slice() }; moved = false; });
    window.addEventListener("pointermove", function (ev) {
      if (!dragging) { return; }
      var rect = canvas.getBoundingClientRect();
      var dx = (ev.clientX - dragging.x) / rect.width * box[2], dy = (ev.clientY - dragging.y) / rect.height * box[3];
      if (Math.abs(ev.clientX - dragging.x) + Math.abs(ev.clientY - dragging.y) > 4) { moved = true; canvas.classList.add("dragging"); }
      box = [dragging.box[0] - dx, dragging.box[1] - dy, box[2], box[3]];
      apply();
    });
    window.addEventListener("pointerup", function () { dragging = null; canvas.classList.remove("dragging"); setTimeout(function () { moved = false; }, 0); });

    function control(label, text, action) { var b = el("button", { type: "button", "aria-label": label, text: text }); b.addEventListener("click", action); return b; }
    var center = function () { return [box[0] + box[2] / 2, box[1] + box[3] / 2]; };
    fill(view,
      el("h1", { class: "section-title", text: "Concept map" }),
      el("p", { class: "section-lede", text: "Each concept sits under the broader one it belongs to. Dashed lines lead from what to learn first. Hover to preview, click to open; drag to move, scroll to zoom." }),
      el("div", { class: "map-wrap" }, [
        canvas,
        tip,
        el("div", { class: "map-controls" }, [
          control("Zoom in", "+", function () { var c = center(); zoom(1 / 1.25, c[0], c[1]); }),
          control("Zoom out", "−", function () { var c = center(); zoom(1.25, c[0], c[1]); }),
          control("Reset view", "↺", function () { box = home.slice(); apply(); })
        ])
      ]),
      el("p", { class: "map-legend", text: "Larger circles have more evidence. Dashed circles were mentioned but not researched yet." })
    );
  }

  // Read, revise, sources -------------------------------------------------------------------------

  function readView() {
    var articles = data.path.map(function (id) {
      var c = concept(id), level = shownLevel(c);
      return el("article", { id: "read-" + id }, [
        el("h2", {}, [el("a", { href: link(id), text: c.title })]),
        level ? el("div", { class: "prose", html: c.levels[level] }) : (c.summaryHtml ? el("p", { html: c.summaryHtml }) : null)
      ]);
    });
    fill(view,
      el("h1", { class: "section-title", text: "Read it all" }),
      el("p", { class: "section-lede", text: "Every concept in learning order, as one article at your chosen depth." }),
      depthControl(null, readView),
      el("div", { class: "reader" }, [
        el("nav", { "aria-label": "Concepts" }, [el("ol", {}, data.path.map(function (id) {
          var a = el("a", { href: "#/read", text: concept(id).title });
          a.addEventListener("click", function (ev) { ev.preventDefault(); document.getElementById("read-" + id).scrollIntoView({ behavior: "smooth" }); });
          return el("li", {}, [a]);
        }))]),
        el("div", {}, articles.length ? articles : [el("p", { text: "Nothing to read yet." })])
      ])
    );
    decorate(view);
  }

  function reviseView() {
    var cards = data.path.map(function (id) {
      var c = concept(id);
      var back = c.levels.summary || c.summaryHtml;
      var card = el("button", { type: "button", class: "card", "aria-pressed": "false" }, [
        el("span", { class: "card-front", text: c.title }),
        el("small", { text: "Recall it, then click to check" })
      ]);
      card.addEventListener("click", function (ev) {
        if (ev.target.closest(".cite")) { return; }
        var flipped = card.getAttribute("aria-pressed") === "true";
        card.setAttribute("aria-pressed", String(!flipped));
        fill.apply(null, [card].concat(flipped
          ? [el("span", { class: "card-front", text: c.title }), el("small", { text: "Recall it, then click to check" })]
          : [el("span", { class: "card-back", html: back || "No summary yet." }), el("small", { text: c.title })]));
        decorate(card);
      });
      return card;
    });
    var verified = Object.keys(data.claims).filter(function (id) { return data.claims[id].status === "verified"; });
    var facts = (verified.length ? verified : Object.keys(data.claims)).slice(0, 40).map(function (id) {
      var claim = data.claims[id];
      return el("li", {}, [el("span", { html: claim.textHtml }), " ", el("button", { type: "button", class: "cite", "data-claim": id, "data-status": claim.status, text: String(claim.number) })]);
    });
    fill(view,
      el("h1", { class: "section-title", text: "Revise" }),
      el("p", { class: "section-lede", text: "Test yourself: name what each concept is, then flip the card." }),
      el("div", { class: "cards" }, cards),
      el("h2", { class: "section-title", text: verified.length ? "Facts confirmed by independent sources" : "Key facts" }),
      el("p", { class: "section-lede", text: verified.length ? "The claims at least two independent sources agree on." : "None are confirmed by two independent sources yet; these rest on one source each." }),
      el("ul", { class: "key-facts" }, facts)
    );
    decorate(view);
  }

  function sourcesView(focus) {
    var plan = data.agenda.map(function (item) {
      return el("li", { "data-status": item.status }, [
        el("span", { class: "plan-status", text: item.status === "done" ? "Covered" : item.status === "dropped" ? "Dropped" : "Open" }),
        el("span", {}, [item.text, item.noteHtml ? el("span", { class: "plan-note", html: item.noteHtml }) : null])
      ]);
    });
    var sources = data.sources.map(function (s) {
      return el("li", { id: "source-" + s.number }, [
        el("span", { class: "source-n", text: String(s.number) }),
        el("div", {}, [
          el("a", { class: "source-title", href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.title }),
          el("p", { class: "source-meta", text: s.tier + " from " + s.domain + (s.published ? ", published " + s.published : "") + ", read " + s.fetched + ". " + (s.claims.length ? "Evidence for " + s.claims.length + " claim" + (s.claims.length === 1 ? "" : "s") + "." : "Not used as evidence.") + (s.duplicateOf ? " Same content as source " + s.duplicateOf + ", so not counted as independent." : "") })
        ])
      ]);
    });
    var runs = data.runs.map(function (r) {
      return el("tr", {}, [el("td", { text: r.started }), el("td", { text: r.status }), el("td", { class: "num", text: String(r.steps) }), el("td", { class: "num", text: r.tokens.toLocaleString() }), el("td", { class: "num", text: r.cost })]);
    });
    fill(view,
      el("h1", { class: "section-title", text: "Sources and method" }),
      el("p", { class: "section-lede", text: "Every page the agent read. Claims are only recorded with a passage quoted from one of these, checked word for word." }),
      el("ol", { class: "source-list" }, sources.length ? sources : [el("li", { text: "No sources yet." })]),
      plan.length ? el("h2", { class: "section-title", text: "Research plan" }) : null,
      plan.length ? el("ul", { class: "plan" }, plan) : null,
      runs.length ? el("h2", { class: "section-title", text: "Research log" }) : null,
      runs.length ? el("div", { class: "table-wrap" }, [el("table", { class: "runs" }, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "Started (UTC)" }), el("th", { text: "Outcome" }), el("th", { class: "num", text: "Steps" }), el("th", { class: "num", text: "Tokens" }), el("th", { class: "num", text: "Cost" })])]),
        el("tbody", {}, runs)
      ])]) : null,
      footer()
    );
    decorate(view);
    if (focus) {
      var target = document.getElementById("source-" + focus);
      if (target) { target.scrollIntoView({ block: "center" }); target.setAttribute("tabindex", "-1"); target.focus({ preventScroll: true }); }
    }
  }

  function notFound() {
    fill(view, el("h1", { class: "section-title", text: "Not found" }), el("p", {}, [el("a", { href: "#/", text: "Back to the overview" })]));
  }

  function footer() {
    return el("footer", { class: "footer", text: "Generated by ResearchOS on " + data.generatedAt + " from project " + data.projectId + "." });
  }

  // Router ----------------------------------------------------------------------------------------

  function route() {
    hideEvidence();
    var parts = location.hash.replace(/^#\/?/, "").split("/");
    var name = parts[0] || "";
    var current = { "": null, learn: "learn", concept: "learn", map: "map", read: "read", revise: "revise", sources: "sources" }[name];
    document.querySelectorAll(".bar-nav a").forEach(function (a) {
      if (a.getAttribute("data-view") === current) { a.setAttribute("aria-current", "page"); } else { a.removeAttribute("aria-current"); }
    });
    if (name === "concept") { conceptView(decodeURIComponent(parts[1] || "")); }
    else if (name === "learn") { var start = load(KEY + "last"); var to = start && concept(start) ? start : data.path[0]; if (to) { location.replace(link(to)); return; } overview(); }
    else if (name === "map") { mapView(); }
    else if (name === "read") { readView(); }
    else if (name === "revise") { reviseView(); }
    else if (name === "sources") { sourcesView(parts[1]); }
    else if (name === "") { overview(); }
    else { notFound(); }
    if (name !== "sources" || !parts[1]) { window.scrollTo(0, 0); }
    view.focus({ preventScroll: true });
  }
  window.addEventListener("hashchange", route);

  // Search ------------------------------------------------------------------------------------------

  var dialog = document.getElementById("search");
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var index = [];
  Object.keys(data.concepts).forEach(function (id) {
    var c = data.concepts[id];
    index.push({ kind: "Concept", title: c.title, text: plain(c.summaryHtml), href: link(id) });
  });
  Object.keys(data.claims).forEach(function (id) {
    var claim = data.claims[id];
    var target = claim.concepts[0] ? link(claim.concepts[0]) : "#/revise";
    index.push({ kind: "Claim " + claim.number + " · " + claim.statusLabel, title: plain(claim.textHtml), text: "", href: target });
  });
  data.sources.forEach(function (s) { index.push({ kind: "Source " + s.number, title: s.title, text: s.domain, href: "#/sources/" + s.number }); });
  index.forEach(function (item) { item.hay = (item.title + " " + item.text).toLowerCase(); });

  function highlight(parent, text, words) {
    if (!words.length) { parent.textContent = text; return; }
    var pattern = new RegExp("(" + words.map(function (w) { return w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }).join("|") + ")", "gi");
    var last = 0;
    text.replace(pattern, function (match, _g, offset) {
      parent.appendChild(document.createTextNode(text.slice(last, offset)));
      parent.appendChild(el("mark", { text: match }));
      last = offset + match.length;
      return match;
    });
    parent.appendChild(document.createTextNode(text.slice(last)));
  }

  var selected = -1;
  function search() {
    var words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    results.replaceChildren();
    selected = -1;
    if (!words.length) { return; }
    var hits = index.filter(function (item) { return words.every(function (w) { return item.hay.indexOf(w) >= 0; }); }).slice(0, 14);
    if (!hits.length) { results.appendChild(el("p", { text: "Nothing matches “" + input.value.trim() + "”." })); return; }
    hits.forEach(function (hit) {
      var title = el("strong"), text = el("span");
      highlight(title, hit.title.length > 160 ? hit.title.slice(0, 158) + "…" : hit.title, words);
      if (hit.text) { highlight(text, hit.text.length > 160 ? hit.text.slice(0, 158) + "…" : hit.text, words); }
      results.appendChild(el("a", { href: hit.href }, [el("small", { text: hit.kind }), title, hit.text ? text : null]));
    });
  }
  function openSearch() { dialog.hidden = false; input.value = ""; results.replaceChildren(); input.focus(); }
  function closeSearch() { dialog.hidden = true; }
  input.addEventListener("input", search);
  input.addEventListener("keydown", function (ev) {
    var links = results.querySelectorAll("a");
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (!links.length) { return; }
      selected = (selected + (ev.key === "ArrowDown" ? 1 : -1) + links.length) % links.length;
      links.forEach(function (a, i) { a.setAttribute("aria-selected", String(i === selected)); });
      links[selected].scrollIntoView({ block: "nearest" });
    } else if (ev.key === "Enter" && links.length) {
      ev.preventDefault(); links[Math.max(selected, 0)].click();
    }
  });
  results.addEventListener("click", function (ev) { if (ev.target.closest("a")) { closeSearch(); } });
  dialog.addEventListener("click", function (ev) { if (ev.target === dialog) { closeSearch(); } });
  document.getElementById("search-open").addEventListener("click", openSearch);
  document.addEventListener("keydown", function (ev) {
    var typing = /^(input|textarea|select)$/i.test(ev.target.tagName) || ev.target.isContentEditable;
    if (ev.key === "/" && !typing && !ev.metaKey && !ev.ctrlKey && !ev.altKey) { ev.preventDefault(); openSearch(); }
    else if (ev.key === "Escape") { if (!dialog.hidden) { closeSearch(); } else if (!drawer.hidden) { hideEvidence(); } }
  });

  // Theme -----------------------------------------------------------------------------------------------

  var THEMES = ["system", "light", "dark"];
  var themeButton = document.getElementById("theme");
  var theme = THEMES.indexOf(load("researchos-theme")) >= 0 ? load("researchos-theme") : "system";
  function applyTheme() {
    if (theme === "system") { root.removeAttribute("data-theme"); } else { root.setAttribute("data-theme", theme); }
    themeButton.textContent = { system: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[theme];
  }
  themeButton.addEventListener("click", function () { theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]; save("researchos-theme", theme); applyTheme(); });
  applyTheme();

  route();
})();
