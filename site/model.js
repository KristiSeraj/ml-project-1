/* The model, in the browser.
 *
 * `model.json` carries the fitted preprocessor's numbers and the decision
 * tree exported by site/export.py. This file rebuilds the feature vector in
 * the same column order scikit-learn's ColumnTransformer produces — the
 * scaled numeric columns first, then one one-hot block per categorical
 * column — and walks the tree. There is no ML runtime and no network call.
 *
 * Loaded by index.html in the browser, and required by parity.js under node,
 * so the thing the page runs is the thing the parity check tests.
 */

(function (root) {
  "use strict";

  function isMissing(v) {
    return v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v));
  }

  /* The feature vector, in the transform's own column order. */
  function buildVector(model, row) {
    var x = [];
    var num = model.numeric;
    for (var i = 0; i < num.columns.length; i++) {
      var raw = row.numeric[num.columns[i]];
      var value = isMissing(raw) ? num.impute[i] : Number(raw);
      x.push((value - num.mean[i]) / num.scale[i]);
    }

    var cat = model.categorical;
    for (var j = 0; j < cat.columns.length; j++) {
      var given = row.categorical[cat.columns[j]];
      var level = isMissing(given) ? cat.impute[j] : String(given);
      var levels = cat.categories[j];
      for (var k = 0; k < levels.length; k++) {
        // handle_unknown="ignore": a level never seen in May leaves the
        // whole block at zero, exactly as scikit-learn does.
        x.push(levels[k] === level ? 1 : 0);
      }
    }
    return x;
  }

  /* One decision tree: follow x[feature] <= threshold to the left. */
  function walkTree(tree, x) {
    var node = 0;
    while (tree.children_left[node] !== -1) {
      node = x[tree.feature[node]] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
    }
    return tree.value[node];
  }

  function predict(model, row) {
    return walkTree(model.tree, buildVector(model, row));
  }

  /* How many May trips ended in the leaf this prediction came from — the
   * honest way to say "this estimate rests on very little". */
  function leafSupport(model, row) {
    var x = buildVector(model, row);
    var tree = model.tree;
    var node = 0;
    while (tree.children_left[node] !== -1) {
      node = x[tree.feature[node]] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
    }
    return tree.n_node_samples[node];
  }

  /* The planned route, estimated the only way zones allow: centre to centre
   * in a straight line, times a detour factor learned on May. */
  function estimateDistance(zonesFile, pickupId, dropoffId) {
    var byId = {};
    zonesFile.zones.forEach(function (z) { byId[z.id] = z; });
    var a = byId[pickupId];
    var b = byId[dropoffId];
    if (!a || !b) return null;
    if (pickupId === dropoffId) {
      return { miles: zonesFile.detour.same_zone_miles, sameZone: true, straight: 0 };
    }
    var feetPerMile = 5280;
    var straight = Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)) / feetPerMile;
    return {
      miles: straight * zonesFile.detour.factor,
      straight: straight,
      sameZone: false
    };
  }

  var api = {
    buildVector: buildVector,
    predict: predict,
    leafSupport: leafSupport,
    estimateDistance: estimateDistance
  };

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.TripModel = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
