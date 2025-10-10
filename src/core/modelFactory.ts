import * as dotenv from "dotenv";
import { ILLMProvider } from "./providers/ILLMProvider";
import { OpenAIProvider } from "./providers/OpenAIProvider";
import { AnthropicProvider } from "./providers/AnthropicProvider";
import { GoogleGenAIProvider } from "./providers/GoogleGenAIProvider";
import {
    LLMConfig,
    LLMProvider,
    OpenAIConfig,
    AnthropicConfig,
    GoogleGenAIProviderConfig,
} from "./types";

// Load environment variables from .env file
dotenv.config();

export const DEFAULT_TEMPERATURE = 0.7;

/**
 * Type guard for OpenAI config
 */
export function isOpenAIConfig(config: LLMConfig): config is OpenAIConfig {
    return config.provider === "openai";
}

/**
 * Type guard for Anthropic config
 */
export function isAnthropicConfig(
    config: LLMConfig
): config is AnthropicConfig {
    return config.provider === "anthropic";
}

/**
 * Creates an instance of an LLM provider based on the provided configuration
 * @param config The LLM configuration
 * @returns An LLM provider instance
 * @throws Error if the provider is not supported
 */
export function createProvider(config: LLMConfig): ILLMProvider {
    switch (config.provider) {
        case "genai":
            return new GoogleGenAIProvider(config as any); // Temporarily cast to any
        case "openai":
            return new OpenAIProvider((config as OpenAIConfig).apiKey);
        case "anthropic":
            return new AnthropicProvider((config as AnthropicConfig).apiKey);
        default:
            throw new Error(
                `Provider ${config.provider} not supported. Use 'openai' or 'anthropic'.`
            );
    }
}

/**
 * Get a standardized environment variable name for the API key of a provider
 * @param provider The LLM provider
 * @returns The environment variable name
 */
export function getApiKeyEnvVar(provider: LLMProvider): string {
    switch (provider) {
        case "openai":
            return "OPENAI_API_KEY";
        case "anthropic":
            return "ANTHROPIC_API_KEY";
        default:
            return `${provider.toUpperCase()}_API_KEY`;
    }
}
