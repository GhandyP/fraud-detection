/**
 * Illustrative synthetic records, not real transactions.
 * Feature names come from the loaded artifact; documented creditcard features
 * (Time, V1..V28, Amount) are only the API's usual fallback vocabulary.
 */
export function buildExamples(
  featureNames: string[],
): Record<string, number>[] {
  const names = [...new Set(featureNames)];
  return [0, 1, 2].map((exampleIndex) =>
    Object.fromEntries(
      names.map((name, featureIndex) => [
        name,
        Number(
          (((featureIndex + 1) * (exampleIndex + 2)) % 17) / 10 - 0.8,
        ),
      ]),
    ),
  );
}
