/* Does the browser model agree with scikit-learn?
 *
 *   node site/parity.js
 *
 * Compares this repository's JavaScript predictions against the scikit-learn
 * predictions for 2,000 June trips, written by site/export.py. A page that
 * quietly disagrees with the notebooks would make every number on it a lie,
 * so this is the check that keeps them honest.
 */

const fs = require("fs");
const path = require("path");
const { predict } = require("./model.js");

const here = __dirname;
const model = JSON.parse(fs.readFileSync(path.join(here, "model.json"), "utf8"));
const sample = JSON.parse(fs.readFileSync(path.join(here, "parity-sample.json"), "utf8"));

const numericCols = model.numeric.columns;
const categoricalCols = model.categorical.columns;

let worst = 0;
let worstRow = null;

sample.rows.forEach((row, i) => {
  const input = { numeric: {}, categorical: {} };
  numericCols.forEach((c, j) => { input.numeric[c] = row.numeric[j]; });
  categoricalCols.forEach((c, j) => { input.categorical[c] = row.categorical[j]; });

  const mine = predict(model, input);
  const diff = Math.abs(mine - row.sklearn);
  if (diff > worst) { worst = diff; worstRow = { i, mine, sklearn: row.sklearn }; }
});

console.log(`compared ${sample.rows.length} June trips`);
console.log(`largest absolute difference: ${worst.toExponential(3)} minutes`);

if (worst > 1e-6) {
  console.error("PARITY FAILED", worstRow);
  process.exit(1);
}
console.log("parity ok — the page and the notebooks agree");
