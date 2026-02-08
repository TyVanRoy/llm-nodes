import { StreamNode } from "../src/index";

export default async function main() {
    const writer = new StreamNode<{ topic: string }>({
        promptTemplate: "Write a short poem about {{topic}}",
        llmConfig: {
            provider: "anthropic",
            model: "claude-haiku-4-5",
            maxTokens: 1024,
        },
    });

    // Streaming usage
    console.log("--- Streaming ---");
    for await (const chunk of writer.stream({ topic: "the ocean" })) {
        process.stdout.write(chunk.text);
        if (chunk.tokenUsage) {
            console.log("\nTokens:", chunk.tokenUsage);
        }
    }

    // Non-streaming still works (inherited execute())
    console.log("\n--- Non-streaming ---");
    const full = await writer.execute({ topic: "the ocean" });
    console.log(full);
    console.log("Usage:", writer.getTotalTokenUsage());
}
