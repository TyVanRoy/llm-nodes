import { GeneralNodeOptions, StreamChunk, TokenUsage } from "../core/types";
import { LLMNode } from "../core/LLMNode";
import { textParser } from "../parsers/structured";

/**
 * StreamNode
 *
 * A specialized LLMNode that exposes a `stream()` method returning an
 * AsyncGenerator of StreamChunk objects. Each chunk carries incremental
 * text; the final chunk carries tokenUsage with text: "".
 *
 * The inherited `execute()` method still works for non-streaming usage.
 */
export class StreamNode<TInput> extends LLMNode<TInput, string> {
    constructor(options: GeneralNodeOptions<TInput, string>) {
        super({
            ...options,
            parser: textParser(),
        });
    }

    /**
     * Stream the LLM response, yielding incremental text chunks.
     * The final chunk has `text: ""` and carries `tokenUsage`.
     */
    async *stream(input: TInput): AsyncGenerator<StreamChunk> {
        if (!this.provider.invokeStream) {
            throw new Error(
                `Provider '${this.llmConfig.provider}' does not support streaming.`
            );
        }

        const prompt = this.generatePrompt(input);
        let finalUsage: TokenUsage | undefined;

        for await (const chunk of this.provider.invokeStream(prompt, this.llmConfig)) {
            if (chunk.tokenUsage) finalUsage = chunk.tokenUsage;
            yield chunk;
        }

        if (finalUsage) this.recordUsage(finalUsage);
    }
}
