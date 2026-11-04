// Import only MiniLM's BERT architecture from the pinned Transformers.js source.
// The generic pipeline entry registers many unrelated generative/vision models.
import { BertModel } from '@minilm-source/models/bert/modeling_bert.js';
import { BertTokenizer } from '@minilm-source/models/bert/tokenization_bert.js';
import { MODEL_CLASS_TO_NAME_MAPPING, MODEL_TYPE_MAPPING, MODEL_TYPES } from '@minilm-source/models/modeling_utils.js';
import { mean_pooling } from '@minilm-source/utils/tensor.js';
import { env } from '@minilm-source/env.js';

MODEL_CLASS_TO_NAME_MAPPING.set(BertModel, 'BertModel');
MODEL_TYPE_MAPPING.set('BertModel', MODEL_TYPES.EncoderOnly);
MODEL_TYPE_MAPPING.set('bert', MODEL_TYPES.EncoderOnly);
env.allowLocalModels = false;

export async function createMiniLMEncoder(modelId, options) {
  const tokenizer = await BertTokenizer.from_pretrained(modelId);
  const model = await BertModel.from_pretrained(modelId, options);
  return async (text, { max_length = 256 } = {}) => {
    const inputs = tokenizer(text, { padding: true, truncation: true, max_length });
    const outputs = await model(inputs);
    return mean_pooling(outputs.last_hidden_state, inputs.attention_mask).normalize(2, -1);
  };
}
