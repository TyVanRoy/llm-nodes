/**
 * Token usage information from an LLM call
 */
export type TokenUsage = {
    inputTokens: number;
    outputTokens: number;
    thinkingTokens?: number; // For tracking reasoning/thinking tokens separately
    searchCount?: number; // For web search usage tracking
    fetchCount?: number; // For web fetch usage tracking
};

/**
 * Record of a single LLM call usage
 */
export type UsageRecord = {
    timestamp: Date;
    provider: string;
    model: string;
    tokenUsage: TokenUsage;
};

/**
 * Interface for any component that can execute with input and produce output
 */
export interface IExecutable<TInput, TOutput> {
    execute(input: TInput): Promise<TOutput>;
}

/**
 * Supported LLM providers
 */
export type LLMProvider = "openai" | "anthropic" | "grok" | "ollama" | string;

/**
 * Web search configuration
 */
export interface WebSearchConfig {
    enabled: boolean;
    maxUses?: number; // Anthropic only
    allowedDomains?: string[]; // Anthropic only
    userLocation?: string; // Anthropic only
}

/**
 * Web fetch configuration
 */
export interface WebFetchConfig {
    enabled: boolean;
    maxUses?: number; // Anthropic only
    allowedDomains?: string[]; // Anthropic only
    citations?: {
        enabled: boolean;
    }; // Anthropic only
}

/**
 * Base configuration options common to all LLM providers
 */
export interface BaseLLMConfig {
    provider: LLMProvider;
    model: string;
    temperature?: number;
    maxTokens?: number;
    providerOptions?: {
        systemPrompt?: string;
        [key: string]: any;
    };
}

/**
 * OpenAI-specific configuration options
 */
export interface OpenAIConfig extends BaseLLMConfig {
    provider: "openai";
    apiKey?: string;
    organization?: string;
    frequencyPenalty?: number;
    presencePenalty?: number;
    topP?: number;
    reasoning?: {
        effort: "low" | "medium" | "high";
    };
    webSearch?: WebSearchConfig;
    tools?: any[]; // For future tool support
}

/**
 * Anthropic-specific configuration options
 */
export interface AnthropicConfig extends BaseLLMConfig {
    provider: "anthropic";
    apiKey?: string;
    topK?: number;
    topP?: number;
    thinking?: {
        type: "enabled";
        budget_tokens: number; // Min 1024
    };
    webSearch?: WebSearchConfig;
    webFetch?: WebFetchConfig;
    stream?: boolean; // Streaming flag for large responses
}

/**
 * AWS Bedrock-specific configuration options
 * Uses Anthropic models via AWS Bedrock
 */
export interface BedrockConfig extends BaseLLMConfig {
    provider: "bedrock";
    awsRegion?: string;
    awsAccessKeyId?: string;
    awsSecretAccessKey?: string;
    awsSessionToken?: string;
    topK?: number;
    topP?: number;
    thinking?: {
        type: "enabled";
        budget_tokens: number; // Min 1024
    };
    stream?: boolean; // Streaming flag for large responses
}

/**
 * Grok-specific configuration options
 */
export interface GrokConfig extends BaseLLMConfig {
    provider: "grok";
    apiKey?: string;
    topP?: number;
}

/**
 * Google Generative AI configuration options
 */
export interface GoogleGenAIProviderConfig extends BaseLLMConfig {
    provider: "genai";
    apiKey?: string;
    thinking?: {
        type: "enabled" | "disabled";
        budget_tokens?: number;
    };
    topK?: number;
    topP?: number;
}

/**
 * Ollama-specific configuration options
 */
export interface OllamaConfig extends BaseLLMConfig {
    provider: "ollama";
    baseUrl?: string;
    format?: string;
    keepAlive?: string;
    numKeep?: number;
}

/**
 * Fallback config for any other provider
 */
export interface OtherProviderConfig extends BaseLLMConfig {
    [key: string]: any;
}

/**
 * Union type of all supported LLM configurations
 * This is a discriminated union - TypeScript will enforce provider-specific fields
 * based on the provider property value
 */
export type LLMConfig =
    | OpenAIConfig
    | AnthropicConfig
    | BedrockConfig
    | GrokConfig
    | GoogleGenAIProviderConfig
    | OllamaConfig
    | OtherProviderConfig;

/**
 * Helper type to extract config for a specific provider
 */
export type ConfigForProvider<P extends LLMProvider> = P extends "openai"
    ? OpenAIConfig
    : P extends "anthropic"
    ? AnthropicConfig
    : P extends "bedrock"
    ? BedrockConfig
    : P extends "grok"
    ? GrokConfig
    : P extends "ollama"
    ? OllamaConfig
    : OtherProviderConfig;

/**
 * Unified batch status across providers
 */
export type BatchStatus =
    | 'validating'   // OpenAI: input file being validated
    | 'in_progress'  // Both: batch is processing
    | 'finalizing'   // OpenAI: results being prepared
    | 'completed'    // Both: done (Anthropic maps 'ended' → 'completed')
    | 'failed'       // Both: batch-level failure
    | 'expired'      // Both: 24h window exceeded
    | 'cancelling'   // Both: cancel in progress
    | 'cancelled';   // Both: cancelled

/**
 * Serializable metadata returned by createBatch.
 * The caller is responsible for persisting this (DB, Redis, file, etc.)
 * and passing it back to retrieveBatch later.
 */
export interface BatchMetadata {
    batchId: string;
    provider: string;
    model: string;
    requestCount: number;
    createdAt: string; // ISO timestamp
}

/**
 * Overall batch result returned by retrieveBatch
 */
export interface BatchResult<TOutput> {
    status: BatchStatus;
    results?: BatchItemResult<TOutput>[]; // Present when status is 'completed'
    requestCounts?: {
        total: number;
        completed: number;
        failed: number;
        expired?: number;
        cancelled?: number;
    };
}

/**
 * Per-item result within a batch
 */
export interface BatchItemResult<TOutput> {
    index: number;             // Original input array index
    status: 'success' | 'failed' | 'expired' | 'cancelled';
    output?: TOutput;          // Parsed through node's parser
    rawOutput?: string;        // Raw LLM response text
    error?: string;            // Error message if failed
    tokenUsage?: TokenUsage;   // Per-item token usage
}

/**
 * A prompt template, either as a string with variables or a function
 */
export type PromptTemplate<TInput> = string | ((input: TInput) => string);

/**
 * A function that parses the raw LLM output into a structured format
 */
export type ResponseParser<TOutput> = (rawResponse: string) => TOutput;

/**
 * Configuration options for all LLM nodes
 */
export type GeneralNodeOptions<TInput, TOutput> = {
    promptTemplate: PromptTemplate<TInput>;
    llmConfig: LLMConfig;
    inputPreprocessor?: (input: TInput) => any;
};

/**
 * Configuration options for an LLMNode
 */
export type BaseNodeOptions<TInput, TOutput> = {
    parser: ResponseParser<TOutput>;
} & GeneralNodeOptions<TInput, TOutput>;
