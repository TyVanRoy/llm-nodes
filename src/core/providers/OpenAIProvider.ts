import OpenAI, { toFile } from "openai";
import { ILLMProvider, LLMResponse, ProviderBatchRequest, ProviderBatchResponse, ProviderBatchItemResult } from "./ILLMProvider";
import { OpenAIConfig, BatchMetadata, BatchStatus, LLMConfig, StreamChunk, TokenUsage } from "../types";

/**
 * OpenAI provider implementation
 */
export class OpenAIProvider implements ILLMProvider {
    private client: OpenAI;
    readonly provider = "openai";

    constructor(apiKey?: string) {
        this.client = new OpenAI({
            apiKey: apiKey || process.env.OPENAI_API_KEY,
        });
    }

    async invoke(prompt: string, config: OpenAIConfig): Promise<LLMResponse> {
        const {
            model,
            temperature,
            maxTokens,
            topP,
            frequencyPenalty,
            presencePenalty,
            reasoning,
            webSearch,
            providerOptions,
        } = config;

        // Determine API based on model
        const useResponsesAPI = this.shouldUseResponsesAPI(model);

        if (useResponsesAPI) {
            // Use responses API for newer models (GPT-5, etc.)
            const params: any = {
                model,
                input: prompt,
                max_output_tokens: maxTokens,
            };

            // Add optional parameters
            if (providerOptions?.systemPrompt) {
                params.instructions = providerOptions.systemPrompt;
            }
            if (temperature !== undefined) params.temperature = temperature;
            if (topP !== undefined) params.top_p = topP;
            if (frequencyPenalty !== undefined)
                params.frequency_penalty = frequencyPenalty;
            if (presencePenalty !== undefined)
                params.presence_penalty = presencePenalty;

            // Add reasoning if provided (for GPT-5)
            if (reasoning) {
                params.reasoning = reasoning;
            }

            // Add web search if enabled
            if (webSearch?.enabled) {
                params.tools = [{ type: "web_search" }];
            }

            try {
                const response = await this.client.responses.create(params);

                // Extract content from response output
                let content = response.output_text;

                return {
                    content,
                    usage: {
                        inputTokens: response.usage?.input_tokens || 0,
                        outputTokens: response.usage?.output_tokens || 0,
                        thinkingTokens:
                            (response.usage as any)?.reasoning_tokens || 0,
                        searchCount: (response.usage as any)?.search_count,
                    },
                    raw: response,
                };
            } catch (error: any) {
                // If responses API fails, fall back to chat completions
                if (error?.status === 404) {
                    return this.useChatCompletions(prompt, config);
                }
                throw error;
            }
        } else {
            return this.useChatCompletions(prompt, config);
        }
    }

    async *invokeStream(
        prompt: string,
        config: OpenAIConfig
    ): AsyncGenerator<StreamChunk> {
        const useResponsesAPI = this.shouldUseResponsesAPI(config.model);

        if (useResponsesAPI) {
            try {
                yield* this.streamResponsesAPI(prompt, config);
                return;
            } catch (error: any) {
                if (error?.status === 404) {
                    yield* this.streamChatCompletions(prompt, config);
                    return;
                }
                throw error;
            }
        }

        yield* this.streamChatCompletions(prompt, config);
    }

    private async *streamChatCompletions(
        prompt: string,
        config: OpenAIConfig
    ): AsyncGenerator<StreamChunk> {
        const {
            model,
            temperature,
            maxTokens,
            topP,
            frequencyPenalty,
            presencePenalty,
            providerOptions,
        } = config;

        const messages: any[] = [];
        if (providerOptions?.systemPrompt) {
            messages.push({ role: "system", content: providerOptions.systemPrompt });
        }
        messages.push({ role: "user", content: prompt });

        const params: any = {
            model,
            messages,
            stream: true,
            stream_options: { include_usage: true },
        };

        if (maxTokens !== undefined) params.max_tokens = maxTokens;
        if (temperature !== undefined) params.temperature = temperature;
        if (topP !== undefined) params.top_p = topP;
        if (frequencyPenalty !== undefined) params.frequency_penalty = frequencyPenalty;
        if (presencePenalty !== undefined) params.presence_penalty = presencePenalty;

        const stream = await this.client.chat.completions.create(params) as unknown as AsyncIterable<any>;

        const tokenUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

        for await (const chunk of stream) {
            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) {
                yield { text: delta };
            }
            if (chunk.usage) {
                tokenUsage.inputTokens = chunk.usage.prompt_tokens || 0;
                tokenUsage.outputTokens = chunk.usage.completion_tokens || 0;
            }
        }

        yield { text: "", tokenUsage };
    }

    private async *streamResponsesAPI(
        prompt: string,
        config: OpenAIConfig
    ): AsyncGenerator<StreamChunk> {
        const {
            model,
            temperature,
            maxTokens,
            topP,
            frequencyPenalty,
            presencePenalty,
            reasoning,
            webSearch,
            providerOptions,
        } = config;

        const params: any = {
            model,
            input: prompt,
            stream: true,
        };

        if (maxTokens !== undefined) params.max_output_tokens = maxTokens;
        if (providerOptions?.systemPrompt) params.instructions = providerOptions.systemPrompt;
        if (temperature !== undefined) params.temperature = temperature;
        if (topP !== undefined) params.top_p = topP;
        if (frequencyPenalty !== undefined) params.frequency_penalty = frequencyPenalty;
        if (presencePenalty !== undefined) params.presence_penalty = presencePenalty;
        if (reasoning) params.reasoning = reasoning;
        if (webSearch?.enabled) params.tools = [{ type: "web_search" }];

        const stream = await this.client.responses.create(params) as unknown as AsyncIterable<any>;

        const tokenUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

        for await (const event of stream) {
            if (event.type === "response.output_text.delta") {
                yield { text: (event as any).delta || "" };
            } else if (event.type === "response.completed") {
                const usage = (event as any).response?.usage;
                if (usage) {
                    tokenUsage.inputTokens = usage.input_tokens || 0;
                    tokenUsage.outputTokens = usage.output_tokens || 0;
                }
            }
        }

        yield { text: "", tokenUsage };
    }

    private async useChatCompletions(
        prompt: string,
        config: OpenAIConfig
    ): Promise<LLMResponse> {
        const {
            model,
            temperature,
            maxTokens,
            topP,
            frequencyPenalty,
            presencePenalty,
            providerOptions,
        } = config;

        // Use chat completions for older models
        const messages: any[] = [];
        if (providerOptions?.systemPrompt) {
            messages.push({
                role: "system",
                content: providerOptions.systemPrompt,
            });
        }
        messages.push({ role: "user", content: prompt });

        const params: any = {
            model,
            messages,
        };

        // Add optional parameters
        if (maxTokens !== undefined) params.max_tokens = maxTokens;
        if (temperature !== undefined) params.temperature = temperature;
        if (topP !== undefined) params.top_p = topP;
        if (frequencyPenalty !== undefined)
            params.frequency_penalty = frequencyPenalty;
        if (presencePenalty !== undefined)
            params.presence_penalty = presencePenalty;

        const response = await this.client.chat.completions.create(params);

        return {
            content: response.choices[0]?.message?.content || "",
            usage: {
                inputTokens: response.usage?.prompt_tokens || 0,
                outputTokens: response.usage?.completion_tokens || 0,
                thinkingTokens:
                    response.usage?.completion_tokens_details
                        ?.reasoning_tokens || 0,
            },
            raw: response,
        };
    }

    private shouldUseResponsesAPI(model: string): boolean {
        // Use responses API for GPT-5 and newer models
        // Note: We'll try responses API first and fall back if needed
        const responsesModels = ["gpt-5", "gpt-4o", "o1", "o3", "o4"];
        return responsesModels.some((m) => model.toLowerCase().includes(m));
    }

    supportsBatch(): boolean {
        return true;
    }

    async createBatch(
        requests: ProviderBatchRequest[],
        config: LLMConfig
    ): Promise<BatchMetadata> {
        const openaiConfig = config as OpenAIConfig;
        const { model, temperature, maxTokens, topP, frequencyPenalty, presencePenalty, reasoning, providerOptions } = openaiConfig;
        const useResponsesAPI = this.shouldUseResponsesAPI(model);
        const endpoint = useResponsesAPI ? "/v1/responses" : "/v1/chat/completions";

        // Build JSONL content
        const lines = requests.map((req) => {
            if (useResponsesAPI) {
                const body: any = { model, input: req.prompt };
                if (maxTokens !== undefined) body.max_output_tokens = maxTokens;
                if (temperature !== undefined) body.temperature = temperature;
                if (topP !== undefined) body.top_p = topP;
                if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
                if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
                if (reasoning) body.reasoning = reasoning;
                if (providerOptions?.systemPrompt) body.instructions = providerOptions.systemPrompt;
                return JSON.stringify({ custom_id: req.customId, method: "POST", url: endpoint, body });
            } else {
                const messages: any[] = [];
                if (providerOptions?.systemPrompt) {
                    messages.push({ role: "system", content: providerOptions.systemPrompt });
                }
                messages.push({ role: "user", content: req.prompt });
                const body: any = { model, messages };
                if (maxTokens !== undefined) body.max_tokens = maxTokens;
                if (temperature !== undefined) body.temperature = temperature;
                if (topP !== undefined) body.top_p = topP;
                if (frequencyPenalty !== undefined) body.frequency_penalty = frequencyPenalty;
                if (presencePenalty !== undefined) body.presence_penalty = presencePenalty;
                return JSON.stringify({ custom_id: req.customId, method: "POST", url: endpoint, body });
            }
        });

        const jsonlContent = lines.join("\n");

        // Upload JSONL file
        const file = await this.client.files.create({
            file: await toFile(Buffer.from(jsonlContent), "batch_input.jsonl"),
            purpose: "batch" as any,
        });

        // Create batch
        const batch = await this.client.batches.create({
            input_file_id: file.id,
            endpoint: endpoint as any,
            completion_window: "24h",
        });

        return {
            batchId: batch.id,
            provider: "openai",
            model,
            requestCount: requests.length,
            createdAt: new Date().toISOString(),
        };
    }

    async retrieveBatch(
        metadata: BatchMetadata,
        config: LLMConfig
    ): Promise<ProviderBatchResponse> {
        const batch = await this.client.batches.retrieve(metadata.batchId);

        const status = batch.status as BatchStatus;

        const requestCounts = {
            total: batch.request_counts?.total || 0,
            completed: batch.request_counts?.completed || 0,
            failed: batch.request_counts?.failed || 0,
        };

        // If not completed, return status only
        if (status !== "completed") {
            return { status, requestCounts };
        }

        const results: ProviderBatchItemResult[] = [];

        // Download and parse output file
        if (batch.output_file_id) {
            const fileResponse = await this.client.files.content(batch.output_file_id);
            const fileContents = await fileResponse.text();
            const outputLines = fileContents.split("\n").filter((line) => line.trim());

            const useResponsesAPI = this.shouldUseResponsesAPI(metadata.model);

            for (const line of outputLines) {
                const entry = JSON.parse(line);
                const itemResult: ProviderBatchItemResult = {
                    customId: entry.custom_id,
                    status: "failed",
                };

                if (entry.response?.status_code === 200) {
                    const body = entry.response.body;
                    let content = "";
                    let inputTokens = 0;
                    let outputTokens = 0;

                    if (useResponsesAPI) {
                        // Responses API format
                        content = body.output_text || "";
                        inputTokens = body.usage?.input_tokens || 0;
                        outputTokens = body.usage?.output_tokens || 0;
                    } else {
                        // Chat completions format
                        content = body.choices?.[0]?.message?.content || "";
                        inputTokens = body.usage?.prompt_tokens || 0;
                        outputTokens = body.usage?.completion_tokens || 0;
                    }

                    itemResult.status = "success";
                    itemResult.content = content;
                    itemResult.tokenUsage = { inputTokens, outputTokens };
                } else if (entry.error) {
                    itemResult.error = entry.error.message || "Request failed";
                } else {
                    itemResult.error = `HTTP ${entry.response?.status_code || "unknown"}`;
                }

                results.push(itemResult);
            }
        }

        // Download and parse error file
        if (batch.error_file_id) {
            const errorResponse = await this.client.files.content(batch.error_file_id);
            const errorContents = await errorResponse.text();
            const errorLines = errorContents.split("\n").filter((line) => line.trim());

            for (const line of errorLines) {
                const entry = JSON.parse(line);
                // Only add if not already in results (output file has successes, error file has failures)
                const existingIndex = results.findIndex((r) => r.customId === entry.custom_id);
                if (existingIndex === -1) {
                    results.push({
                        customId: entry.custom_id,
                        status: entry.error?.code === "batch_expired" ? "expired" : "failed",
                        error: entry.error?.message || "Unknown error",
                    });
                }
            }
        }

        return { status, results, requestCounts };
    }
}
