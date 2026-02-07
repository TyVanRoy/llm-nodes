import { z } from "zod";
import { StructuredOutputNode, BatchMetadata, BatchResult } from "../src/index";

// --- Input & Output Schemas ---

type MovieReviewInput = {
    title: string;
    genre: string;
    yearReleased: number;
};

const MovieAnalysisSchema = z.object({
    sentiment: z.enum(["positive", "negative", "mixed"]),
    themes: z.array(z.string()).min(1).max(5),
    rating: z.number().min(1).max(10),
});

type MovieAnalysis = z.infer<typeof MovieAnalysisSchema>;

// --- Node Definition ---

const movieAnalyzer = new StructuredOutputNode<MovieReviewInput, MovieAnalysis>({
    schema: MovieAnalysisSchema,
    promptTemplate: `Analyze the movie "{{title}}" ({{genre}}, {{yearReleased}}).
Respond with a JSON object containing:
- "sentiment": one of "positive", "negative", or "mixed"
- "themes": an array of 1-5 key themes (strings)
- "rating": a number from 1-10

Respond ONLY with the JSON object with the following schema:
{
  "sentiment": "positive" | "negative" | "mixed",
  "themes": string[],
  "rating": number
}
and no other text.
`,
    llmConfig: {
        provider: "anthropic",
        model: "claude-haiku-4-5",
        maxTokens: 256,
    },
});

// --- Batch Inputs ---

const movies: MovieReviewInput[] = [
    { title: "The Shawshank Redemption", genre: "Drama", yearReleased: 1994 },
    { title: "Inception", genre: "Sci-Fi", yearReleased: 2010 },
    { title: "The Room", genre: "Drama", yearReleased: 2003 },
    { title: "Parasite", genre: "Thriller", yearReleased: 2019 },
    { title: "Mad Max: Fury Road", genre: "Action", yearReleased: 2015 },
    { title: "The Godfather", genre: "Crime", yearReleased: 1972 },
    { title: "Arrival", genre: "Sci-Fi", yearReleased: 2016 },
    { title: "Cats", genre: "Musical", yearReleased: 2019 },
    { title: "Everything Everywhere All at Once", genre: "Sci-Fi", yearReleased: 2022 },
    { title: "Blade Runner 2049", genre: "Sci-Fi", yearReleased: 2017 },
];

// --- Main ---

const POLL_INTERVAL_MS = 10_000;

export default async function main() {
    console.log("=== Batch Processing Example ===\n");
    console.log(`Submitting ${movies.length} movie analysis requests...\n`);

    // --- Create Batch ---
    const metadata: BatchMetadata = await movieAnalyzer.createBatch(movies);
    console.log("Batch created successfully!");
    console.log("Metadata (persist this):");
    console.log(JSON.stringify(metadata, null, 2));
    console.log();

    // --- Poll for Results ---
    let result: BatchResult<MovieAnalysis>;
    let pollCount = 0;

    while (true) {
        pollCount++;
        console.log(`[Poll #${pollCount}] Checking batch status...`);

        result = await movieAnalyzer.retrieveBatch(metadata);

        console.log(`  Status: ${result.status}`);
        if (result.requestCounts) {
            console.log(`  Counts: ${JSON.stringify(result.requestCounts)}`);
        }

        if (result.status === "completed") {
            console.log("  Batch completed!\n");
            break;
        }

        if (result.status === "failed" || result.status === "expired" || result.status === "cancelled") {
            console.error(`  Batch ended with status: ${result.status}`);
            process.exit(1);
        }

        console.log(`  Waiting ${POLL_INTERVAL_MS / 1000}s before next poll...\n`);
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }

    // --- Process Results ---
    console.log("=== Results ===\n");

    for (const item of result!.results!) {
        const movie = movies[item.index];
        console.log(`[${item.index}] "${movie.title}" (${movie.yearReleased})`);
        console.log(`    Status: ${item.status}`);

        if (item.status === "success" && item.output) {
            console.log(`    Sentiment: ${item.output.sentiment}`);
            console.log(`    Rating:    ${item.output.rating}/10`);
            console.log(`    Themes:    ${item.output.themes.join(", ")}`);
        } else {
            console.log(`    Error: ${item.error}`);
            if (item.rawOutput) {
                console.log(`    Raw output: ${item.rawOutput.substring(0, 200)}`);
            }
        }

        if (item.tokenUsage) {
            console.log(`    Tokens:    ${item.tokenUsage.inputTokens} in / ${item.tokenUsage.outputTokens} out`);
        }

        console.log();
    }

    // --- Aggregate Stats ---
    const totalUsage = movieAnalyzer.getTotalTokenUsage();
    console.log("=== Aggregate Token Usage ===");
    console.log(JSON.stringify(totalUsage, null, 2));

    const successCount = result!.results!.filter((r) => r.status === "success").length;
    const failCount = result!.results!.filter((r) => r.status !== "success").length;
    console.log(`\n${successCount} succeeded, ${failCount} failed out of ${movies.length} total`);
}
