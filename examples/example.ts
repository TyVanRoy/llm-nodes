import { TextNode } from "../src/index";

export default async function main() {
    type TextNodeInput = {
        format: string;
        topic: string;
        style: string;
        minWords: number;
        maxWords: number;
    };

    const textGenerator = new TextNode<TextNodeInput>({
        promptTemplate:
            "Search the web for recent information about {{topic}} and write a {{format}} in {{style}} style with {{minWords}} to {{maxWords}} words.",
        llmConfig: {
            provider: "anthropic",
            model: "claude-sonnet-4-5",
            maxTokens: 50000,
            stream: true,
            webSearch: { enabled: true },
        },
    });

    // Use it
    const text = await textGenerator.execute({
        format: "sonnet",
        topic: "ai",
        style: "uplifting",
        minWords: 10,
        maxWords: 150,
    });
    console.log(text);
    console.log(JSON.stringify(textGenerator.getTotalTokenUsage(), null, 2));
}
