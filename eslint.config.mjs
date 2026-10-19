// Lint the layerviz browser modules and their node tests against the repo size rules.
// The scaffold check keeps only findings on lines a change touched.

const sizeRules = {
  complexity: ["error", 10],
  "max-params": ["error", 5],
  "max-lines": ["error", { max: 400, skipBlankLines: false, skipComments: false }],
  "max-lines-per-function": ["error", { max: 50, skipBlankLines: true, skipComments: true }],
};

// Writing markup from strings opens the page to injected script, so build DOM nodes instead.
const safetyRules = {
  "no-eval": "error",
  "no-implied-eval": "error",
  "no-new-func": "error",
  "no-restricted-properties": [
    "error",
    { property: "innerHTML", message: "Build nodes and set textContent instead." },
    { property: "outerHTML", message: "Build nodes and set textContent instead." },
  ],
};

export default [
  { ignores: ["layerviz/web/vendor/", "node_modules/", ".venv/", ".claude/", "runs/"] },
  {
    files: ["**/*.js", "**/*.mjs"],
    languageOptions: { ecmaVersion: "latest", sourceType: "module" },
    rules: { ...sizeRules, ...safetyRules },
  },
  {
    // A test file is one describe block per behavior, so its callbacks run long by design.
    files: ["tests/js/**/*.js"],
    rules: { "max-lines-per-function": "off" },
  },
];
