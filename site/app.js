/* The page: load the three exported files, fill the tables, and predict.
 *
 * Nothing here computes a reported number — every score, buffer and slice
 * comes from results.json, written by site/export.py from the same src/
 * code the notebooks import. The only arithmetic on this page is the
 * prediction itself, and node site/parity.js checks that against
 * scikit-learn.
 */

(function () {
  "use strict";

  var model, zones, results, zoneById = {};

  var el = function (id) { return document.getElementById(id); };
  var fmt1 = function (v) { return v.toFixed(1); };
  var pct = function (v) { return (100 * v).toFixed(0) + "%"; };
  var thousands = function (n) { return n.toLocaleString("en-US"); };

  /* ------------------------------------------------------------- loading */

  Promise.all([
    fetch("model.json").then(function (r) { return r.json(); }),
    fetch("zones.json").then(function (r) { return r.json(); }),
    fetch("results.json").then(function (r) { return r.json(); })
  ]).then(function (loaded) {
    model = loaded[0];
    zones = loaded[1];
    results = loaded[2];
    zones.zones.forEach(function (z) { zoneById[z.id] = z; });

    fillZoneSelects();
    fillResults();
    fillSlices();
    fillHours();
    fillHowItWorks();
    wireUp();
    predict();
  }).catch(function (err) {
    el("caveat").textContent =
      "Could not load the model files. Open this page over HTTP " +
      "(python -m http.server) rather than as a file:// URL. " + err;
  });

  /* ------------------------------------------------------------ the form */

  function fillZoneSelects() {
    var boroughs = {};
    zones.zones.forEach(function (z) {
      (boroughs[z.borough] = boroughs[z.borough] || []).push(z);
    });

    ["pu", "do"].forEach(function (which) {
      var select = el(which);
      Object.keys(boroughs).sort().forEach(function (borough) {
        var group = document.createElement("optgroup");
        group.label = borough;
        boroughs[borough]
          .slice()
          .sort(function (a, b) { return a.zone.localeCompare(b.zone); })
          .forEach(function (z) {
            var option = document.createElement("option");
            option.value = String(z.id);
            option.textContent = z.zone;
            group.appendChild(option);
          });
        select.appendChild(group);
      });
    });

    // Something recognisable and genuinely hard: Midtown to JFK.
    setZone("pu", "Midtown Center", 161);
    setZone("do", "JFK Airport", 132);

    // A weekday rush hour, which is where the model is weakest.
    el("when").value = "2026-06-17T17:30";
  }

  function setZone(which, name, fallbackId) {
    var match = zones.zones.filter(function (z) { return z.zone === name; })[0];
    el(which).value = String(match ? match.id : fallbackId);
  }

  function wireUp() {
    ["pu", "do", "when", "pax"].forEach(function (id) {
      el(id).addEventListener("change", predict);
      el(id).addEventListener("input", predict);
    });
  }

  /* ------------------------------------------------------- the prediction */

  var AIRPORTS = { 1: "EWR", 132: "JFK", 138: "LGA" };

  function predict() {
    var pu = parseInt(el("pu").value, 10);
    var dof = parseInt(el("do").value, 10);
    var when = el("when").value;
    var pax = parseInt(el("pax").value, 10);

    if (!when) { return; }
    var t = new Date(when);
    if (isNaN(t.getTime())) { return; }

    var route = TripModel.estimateDistance(zones, pu, dof);
    if (!route) { return; }

    var dow = (t.getDay() + 6) % 7;           // Monday = 0, as pandas has it
    var row = {
      numeric: {
        trip_distance: route.miles,
        pickup_hour: t.getHours(),
        pickup_dow: dow,
        pickup_minutes_into_day: t.getHours() * 60 + t.getMinutes(),
        passenger_count: isNaN(pax) ? null : pax
      },
      categorical: {
        pickup_borough: zoneById[pu].borough,
        dropoff_borough: zoneById[dof].borough,
        pickup_airport: AIRPORTS[pu] || "none",
        dropoff_airport: AIRPORTS[dof] || "none",
        is_weekend: dow >= 5 ? "weekend" : "weekday"
      }
    };

    var point = TripModel.predict(model, row);
    var buffer = results.buffer_min;
    var support = TripModel.leafSupport(model, row);

    el("point").textContent = fmt1(point) + " min";
    el("buffer").textContent = "+ " + fmt1(buffer) + " min";
    el("shown").textContent = fmt1(point + buffer);
    el("route").textContent = route.sameZone
      ? fmt1(route.miles) + " mi (same zone)"
      : fmt1(route.miles) + " mi";
    el("support").textContent = thousands(support);

    var arrival = new Date(t.getTime() + (point + buffer) * 60000);
    var clock = arrival.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

    var borough = zoneById[pu].borough;
    var slice = results.slices.borough.filter(function (s) { return s.group === borough; })[0];
    var airportTrip = AIRPORTS[pu] || AIRPORTS[dof];

    var note = "Arriving by about " + clock + ". ";
    var warn = false;

    if (slice && slice.coverage_global < results.coverage_target) {
      warn = true;
      note += "But this buffer only gets " + pct(slice.coverage_global) +
        " of " + borough + " pickups there on time, not the " +
        pct(results.coverage_target) + " promised — it needs " +
        fmt1(slice.needs_buffer) + " minutes. ";
    }
    if (airportTrip) {
      warn = true;
      note += "Airport trips are the hardest group in the data: " +
        "the model is out by " +
        fmt1(results.slices.airport.filter(function (s) { return s.group === "True"; })[0].mae) +
        " minutes on average there. ";
    }
    if (support < 50) {
      warn = true;
      note += "Only " + support + " May trips sit in the leaf this estimate came " +
        "from, so treat it as thin. ";
    }
    if (!warn) {
      note += "This trip sits in a group the buffer serves as promised.";
    }

    el("caveat").textContent = note;
    el("caveat").className = warn ? "caveat warn" : "caveat";
  }

  /* ---------------------------------------------------------- the tables */

  function fillResults() {
    var body = el("ladder").querySelector("tbody");
    results.ladder.forEach(function (row, i) {
      var tr = document.createElement("tr");
      if (i === results.ladder.length - 1) { tr.className = "chosen"; }
      [row.name, fmt1(row.mae), fmt1(row.rmse), (row.bias > 0 ? "+" : "") + fmt1(row.bias)]
        .forEach(function (cell, j) {
          var td = document.createElement(j === 0 ? "th" : "td");
          td.textContent = cell;
          tr.appendChild(td);
        });
      body.appendChild(tr);
    });

    var july = el("july").querySelector("tbody");
    [["Chosen model", results.july.model], ["Median baseline", results.july.baseline]]
      .forEach(function (pair) {
        var mae = pair[1].mae;
        var tr = document.createElement("tr");
        tr.innerHTML =
          "<th>" + pair[0] + "</th><td>" + mae[0].toFixed(2) + "</td>" +
          "<td>[" + mae[1].toFixed(2) + ", " + mae[2].toFixed(2) + "]</td>";
        july.appendChild(tr);
      });

    var facts = [
      "Chosen on June: <strong>" + fmt1(results.buffer_min) +
        " minutes</strong> added to every estimate.",
      "On July it delivered <strong>" + pct(results.coverage_july) +
        "</strong> coverage against the " + pct(results.coverage_target) + " promised.",
      "July alone would have needed <strong>" + fmt1(results.buffer_july_would_need) +
        " minutes</strong>, so June's choice is conservative.",
      "With no buffer at all, only <strong>" + pct(results.coverage_no_buffer) +
        "</strong> of passengers arrive before the time shown."
    ];
    el("buffer-facts").innerHTML = facts.map(function (f) {
      return "<li>" + f + "</li>";
    }).join("");
  }

  function fillSlices() {
    ["borough", "distance_band", "airport"].forEach(function (key) {
      var body = el("slice-" + key).querySelector("tbody");
      results.slices[key].forEach(function (s) {
        var missed = s.coverage_global < results.coverage_target;
        var label = key === "airport" ? (s.group === "True" ? "Airport" : "Everything else") : s.group;
        var tr = document.createElement("tr");
        tr.innerHTML =
          "<th>" + label + "</th>" +
          "<td>" + thousands(s.n) + "</td>" +
          "<td>" + fmt1(s.mae) + "</td>" +
          "<td>" + fmt1(s.needs_buffer) + "</td>" +
          '<td class="' + (missed ? "miss" : "") + '">' + pct(s.coverage_global) + "</td>";
        body.appendChild(tr);
      });
    });
  }

  function fillHours() {
    var hours = results.slices.hour.slice().sort(function (a, b) {
      return Number(a.group) - Number(b.group);
    });
    var max = Math.max.apply(null, hours.map(function (h) { return h.mae; }));

    var legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML =
      '<span><i></i>Meets the ' + pct(results.coverage_target) + " promise</span>" +
      '<span><i class="miss"></i>Below it — the app runs late here</span>';

    var wrap = el("hours");
    wrap.parentNode.insertBefore(legend, wrap);

    hours.forEach(function (h) {
      var missed = h.coverage_global < results.coverage_target;
      var bar = document.createElement("div");
      bar.className = "bar" + (missed ? " miss" : "");
      bar.tabIndex = 0;
      bar.setAttribute("data-tip",
        String(h.group).padStart(2, "0") + ":00\n" +
        "MAE " + fmt1(h.mae) + " min\n" +
        thousands(h.n) + " trips\n" +
        "coverage " + pct(h.coverage_global));
      bar.innerHTML =
        '<div class="fill" style="height:' + (100 * h.mae / max).toFixed(1) + '%"></div>' +
        '<div class="tick">' + (Number(h.group) % 3 === 0 ? h.group : "") + "</div>";
      wrap.appendChild(bar);
    });
  }

  function fillHowItWorks() {
    var n = results.n;
    el("split-n").textContent =
      "May " + thousands(n.train) + " · June " + thousands(n.validation) +
      " · July " + thousands(n.test) + " trips";
    el("depth").textContent = model.depth;
    el("nodes").textContent = thousands(model.n_nodes);
    el("rows").textContent = thousands(model.trained_on.rows);
    el("model-size").textContent = "about " +
      Math.round(JSON.stringify(model).length / 1024) + " KB";
    el("detour-factor").textContent = zones.detour.factor + "x";
    el("detour-error").textContent = zones.detour.median_abs_error_mi;
  }
})();
