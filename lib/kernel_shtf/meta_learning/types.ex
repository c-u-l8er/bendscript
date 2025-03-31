defmodule KernelShtf.MetaLearning.Types do
  import KernelShtf.BenBen

  # Define our learning structures
  phrenia Learning do
    model(parameters, heuristics, meta_rules)
    parameter(name, value, update_rate)
    heuristic(condition, recu(action), confidence)
    metaRule(pattern, recu(transformation), priority)
    dataPoint(features, label)
    dataset(data_points, statistics)
    batch(data_subset, batch_size, statistics)
    learningEvidence(patterns, effectiveness, surprise)
  end
end
