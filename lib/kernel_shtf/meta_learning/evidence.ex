defmodule KernelShtf.MetaLearning.Evidence do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning

  # Extract learning evidence from dataset for meta-rule adaptation
  def extract_learning_evidence(dataset) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        # Analyze statistics to extract patterns, effectiveness and surprises
        patterns = extract_learning_patterns(statistics)
        effectiveness = calculate_learning_effectiveness(statistics)
        surprise = measure_learning_surprise(statistics)

        Learning.learningEvidence(patterns, effectiveness, surprise)
    end
  end

  # Extract patterns from learning statistics
  def extract_learning_patterns(statistics) do
    # Find repeating patterns in parameter updates
    param_update_keys =
      Enum.filter(Map.keys(statistics), fn
        {:param_updates, _} -> true
        _ -> false
      end)

    # Analyze patterns in parameter updates
    Enum.map(param_update_keys, fn key ->
      {param_name, updates} = {elem(key, 1), Map.get(statistics, key, [])}
      {param_name, analyze_update_pattern(updates)}
    end)
    |> Enum.into(%{})
  end

  # Analyze pattern in a sequence of updates
  def analyze_update_pattern(updates) do
    cond do
      # Check for oscillation pattern
      updates_oscillating?(updates) ->
        %{type: :oscillation, magnitude: calculate_oscillation_magnitude(updates)}

      # Check for convergence pattern
      updates_converging?(updates) ->
        %{type: :convergence, rate: calculate_convergence_rate(updates)}

      # Check for divergence pattern
      updates_diverging?(updates) ->
        %{type: :divergence, rate: calculate_divergence_rate(updates)}

      # Default pattern
      true ->
        %{type: :random, variance: calculate_updates_variance(updates)}
    end
  end

  # Check if updates are oscillating (alternating signs)
  def updates_oscillating?(updates) when length(updates) >= 4 do
    # Check if signs alternate frequently
    signs = Enum.map(updates, fn update -> if update > 0, do: 1, else: -1 end)
    sign_changes = count_sign_changes(signs)

    # High number of sign changes indicates oscillation
    sign_changes >= length(updates) * 0.4
  end

  def updates_oscillating?(_), do: false

  # Count sign changes in a sequence
  def count_sign_changes(signs) do
    Enum.chunk_every(signs, 2, 1, :discard)
    |> Enum.count(fn [a, b] -> a != b end)
  end

  # Calculate magnitude of oscillation
  def calculate_oscillation_magnitude(updates) do
    abs(Enum.sum(updates) / length(updates))
  end

  # Check if updates are converging (decreasing magnitude)
  def updates_converging?(updates) when length(updates) >= 4 do
    magnitudes = Enum.map(updates, &abs/1)

    # Check if average magnitude decreases over time
    first_half = Enum.take(magnitudes, div(length(magnitudes), 2))
    second_half = Enum.drop(magnitudes, div(length(magnitudes), 2))

    Enum.sum(first_half) / length(first_half) >
      Enum.sum(second_half) / length(second_half) * 1.3
  end

  def updates_converging?(_), do: false

  # Calculate convergence rate
  def calculate_convergence_rate(updates) do
    magnitudes = Enum.map(updates, &abs/1)

    if length(magnitudes) >= 2 do
      first = Enum.at(magnitudes, 0)
      last = Enum.at(magnitudes, -1)

      if first > 0 do
        1.0 - last / first
      else
        0.0
      end
    else
      0.0
    end
  end

  # Check if updates are diverging (increasing magnitude)
  def updates_diverging?(updates) when length(updates) >= 4 do
    magnitudes = Enum.map(updates, &abs/1)

    # Check if average magnitude increases over time
    first_half = Enum.take(magnitudes, div(length(magnitudes), 2))
    second_half = Enum.drop(magnitudes, div(length(magnitudes), 2))

    Enum.sum(second_half) / length(second_half) >
      Enum.sum(first_half) / length(first_half) * 1.3
  end

  def updates_diverging?(_), do: false

  # Calculate divergence rate
  def calculate_divergence_rate(updates) do
    magnitudes = Enum.map(updates, &abs/1)

    if length(magnitudes) >= 2 do
      first = Enum.at(magnitudes, 0)
      last = Enum.at(magnitudes, -1)

      if first > 0 do
        last / first - 1.0
      else
        0.0
      end
    else
      0.0
    end
  end

  # Calculate variance of updates
  def calculate_updates_variance(updates) do
    if length(updates) >= 2 do
      mean = Enum.sum(updates) / length(updates)

      Enum.map(updates, fn update -> (update - mean) * (update - mean) end)
      |> Enum.sum()
      |> Kernel./(length(updates))
    else
      0.0
    end
  end

  # Calculate overall learning effectiveness
  def calculate_learning_effectiveness(statistics) do
    # Check for improvement metrics in statistics
    improvement_keys = [
      :validation_error,
      :training_error,
      :accuracy,
      :convergence_rate
    ]

    # Calculate overall effectiveness score
    Enum.reduce(improvement_keys, 0.0, fn key, acc ->
      case Map.get(statistics, key) do
        nil ->
          acc

        value when is_number(value) ->
          acc + normalize_effectiveness_metric(key, value)

        [h | _] = values when is_list(values) and is_number(h) ->
          # Use trend for list of values
          acc + calculate_effectiveness_trend(key, values)

        _ ->
          acc
      end
    end) / length(improvement_keys)
  end

  # Normalize a metric to a 0-1 effectiveness score
  def normalize_effectiveness_metric(key, value) do
    case key do
      :validation_error -> max(0.0, 1.0 - value / 100.0)
      :training_error -> max(0.0, 1.0 - value / 100.0)
      :accuracy -> max(0.0, value)
      :convergence_rate -> max(0.0, min(1.0, value))
      # Default
      _ -> 0.5
    end
  end

  # Calculate effectiveness trend from a series of values
  def calculate_effectiveness_trend(key, values) when length(values) >= 2 do
    # Determine if higher is better for this metric
    higher_is_better =
      case key do
        :validation_error -> false
        :training_error -> false
        :accuracy -> true
        :convergence_rate -> true
        _ -> true
      end

    # Calculate improvement direction
    first = Enum.at(values, 0)
    last = Enum.at(values, -1)

    if higher_is_better do
      # Higher values are better
      cond do
        last > first -> max(0.0, min(1.0, (last - first) / first))
        last == first -> 0.5
        true -> max(0.0, 1.0 - (first - last) / first)
      end
    else
      # Lower values are better
      cond do
        last < first -> max(0.0, min(1.0, (first - last) / first))
        last == first -> 0.5
        true -> max(0.0, 1.0 - (last - first) / first)
      end
    end
  end

  # Default for insufficient data
  def calculate_effectiveness_trend(_key, _values), do: 0.5

  # Measure surprise in learning process
  def measure_learning_surprise(statistics) do
    # Initialize metrics for measuring surprise
    metrics = %{
      unexpected_improvements: detect_unexpected_improvements(statistics),
      sudden_degradations: detect_sudden_degradations(statistics),
      rare_patterns: detect_rare_patterns(statistics),
      outlier_responses: count_outlier_responses(statistics)
    }

    # Calculate overall surprise score
    average_surprise =
      Enum.map(metrics, fn {_key, value} -> value end)
      |> Enum.sum()
      |> Kernel./(map_size(metrics))

    # Return structured surprise information
    %{
      score: average_surprise,
      details: metrics
    }
  end

  # Detect unexpected improvements in learning trends
  def detect_unexpected_improvements(statistics) do
    # Look for metrics that show sudden positive changes
    improvement_keys = [:validation_error, :training_error, :accuracy]

    Enum.reduce(improvement_keys, 0.0, fn key, acc ->
      case Map.get(statistics, key) do
        [h | _] = values when is_list(values) and is_number(h) ->
          # Check for sudden improvements that deviate from the trend
          improvement_surprise = detect_trend_deviation(key, values, :positive)
          acc + improvement_surprise

        _ ->
          acc
      end
    end) / max(1, length(improvement_keys))
  end

  # Detect sudden degradations in performance
  def detect_sudden_degradations(statistics) do
    # Look for metrics that show sudden negative changes
    degradation_keys = [:validation_error, :training_error, :accuracy]

    Enum.reduce(degradation_keys, 0.0, fn key, acc ->
      case Map.get(statistics, key) do
        [h | _] = values when is_list(values) and is_number(h) ->
          # Check for sudden degradations that deviate from the trend
          degradation_surprise = detect_trend_deviation(key, values, :negative)
          acc + degradation_surprise

        _ ->
          acc
      end
    end) / max(1, length(degradation_keys))
  end

  # Detect deviations from expected trends
  def detect_trend_deviation(key, values, direction) when length(values) >= 3 do
    # Determine if higher is better for this metric
    higher_is_better =
      case key do
        :validation_error -> false
        :training_error -> false
        :accuracy -> true
        _ -> true
      end

    # Adjust direction based on whether higher is better
    effective_direction = if higher_is_better, do: direction, else: opposite_direction(direction)

    # Calculate trend line using simple linear regression
    {slope, _intercept} = calculate_trend_line(values)

    # Examine last few values for deviations
    recent_values = Enum.take(values, -3)
    expected_values = predict_with_trend_line(length(recent_values), values)

    # Calculate deviation scores
    deviations =
      Enum.zip(recent_values, expected_values)
      |> Enum.map(fn {actual, expected} ->
        deviation = (actual - expected) / max(0.01, abs(expected))

        # Score is higher if deviation is in the surprising direction
        case effective_direction do
          :positive -> if deviation > 0, do: deviation, else: 0
          :negative -> if deviation < 0, do: -deviation, else: 0
        end
      end)

    # Average the deviation scores, weighting more recent ones higher
    # More recent values have higher weight
    weights = [0.2, 0.3, 0.5]

    if length(deviations) == length(weights) do
      Enum.zip(deviations, weights)
      |> Enum.map(fn {dev, weight} -> dev * weight end)
      |> Enum.sum()
    else
      Enum.sum(deviations) / length(deviations)
    end
  end

  def detect_trend_deviation(_key, _values, _direction), do: 0.0

  # Calculate the opposite direction
  def opposite_direction(:positive), do: :negative
  def opposite_direction(:negative), do: :positive

  # Calculate trend line using simple linear regression
  def calculate_trend_line(values) do
    n = length(values)

    # Generate x coordinates (0, 1, 2, ...)
    x_values = Enum.to_list(0..(n - 1))

    # Calculate sums for regression formula
    sum_x = Enum.sum(x_values)
    sum_y = Enum.sum(values)
    sum_xy = Enum.zip(x_values, values) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
    sum_x_squared = Enum.map(x_values, fn x -> x * x end) |> Enum.sum()

    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n

    {slope, intercept}
  end

  # Predict values based on trend line
  def predict_with_trend_line(count, values) do
    # Calculate trend line
    {slope, intercept} = calculate_trend_line(values)

    # Get starting point for prediction
    start_idx = length(values) - count

    # Generate predictions
    Enum.map(start_idx..(start_idx + count - 1), fn idx ->
      intercept + slope * idx
    end)
  end

  # Detect rare patterns in learning process
  def detect_rare_patterns(statistics) do
    # Look for uncommon sequences in parameter updates
    param_update_keys =
      Enum.filter(Map.keys(statistics), fn
        {:param_updates, _} -> true
        _ -> false
      end)

    Enum.reduce(param_update_keys, 0.0, fn key, acc ->
      updates = Map.get(statistics, key, [])

      # Check for rare patterns in this parameter's updates
      pattern_rarity = calculate_pattern_rarity(updates)

      acc + pattern_rarity
    end) / max(1, length(param_update_keys))
  end

  # Calculate how rare a pattern is in updates
  def calculate_pattern_rarity(updates) when length(updates) >= 5 do
    # Convert updates to a sequence of symbols (e.g., up/down/same)
    symbols =
      Enum.chunk_every(updates, 2, 1, :discard)
      |> Enum.map(fn [a, b] ->
        cond do
          b > a * 1.1 -> :up
          b < a * 0.9 -> :down
          true -> :same
        end
      end)

    # Count frequencies of common patterns
    patterns = [
      # consistent increase
      [:up, :up, :up],
      # consistent decrease
      [:down, :down, :down],
      # oscillation
      [:up, :down, :up],
      # oscillation
      [:down, :up, :down],
      # plateau
      [:same, :same, :same]
    ]

    # Check for each pattern's presence
    pattern_counts =
      Enum.map(patterns, fn pattern ->
        count_pattern(symbols, pattern)
      end)

    # Calculate rarity score - higher when no common patterns are present
    common_pattern_ratio = Enum.sum(pattern_counts) / max(1, length(symbols) - 2)

    1.0 - min(1.0, common_pattern_ratio)
  end

  def calculate_pattern_rarity(_updates), do: 0.0

  # Count occurrences of a pattern in a sequence
  def count_pattern(sequence, pattern) when length(sequence) >= length(pattern) do
    Enum.chunk_every(sequence, length(pattern), 1, :discard)
    |> Enum.count(fn chunk -> chunk == pattern end)
  end

  def count_pattern(_sequence, _pattern), do: 0

  # Count outlier responses in the learning process
  def count_outlier_responses(statistics) do
    # Look for outliers in errors, accuracies, and parameter updates
    outlier_categories = [
      :validation_errors,
      :training_errors,
      :parameter_updates
    ]

    Enum.reduce(outlier_categories, 0.0, fn category, acc ->
      outlier_ratio = calculate_outlier_ratio(statistics, category)
      acc + outlier_ratio
    end) / length(outlier_categories)
  end

  # Calculate ratio of outliers in a category
  def calculate_outlier_ratio(statistics, category) do
    values =
      case category do
        :validation_errors ->
          Map.get(statistics, :validation_error_history, [])

        :training_errors ->
          Map.get(statistics, :training_error_history, [])

        :parameter_updates ->
          # Flatten all parameter updates into one list
          Enum.flat_map(Map.keys(statistics), fn
            {:param_updates, _} = key -> Map.get(statistics, key, [])
            _ -> []
          end)
      end

    # Count outliers using IQR method
    outlier_count = count_outliers(values)

    if length(values) > 0 do
      outlier_count / length(values)
    else
      0.0
    end
  end

  # Count outliers using IQR method
  def count_outliers(values) when length(values) >= 4 do
    # Sort values
    sorted = Enum.sort(values)
    n = length(sorted)

    # Find quartiles
    q1_idx = div(n, 4)
    q3_idx = div(3 * n, 4)

    q1 = Enum.at(sorted, q1_idx)
    q3 = Enum.at(sorted, q3_idx)

    # Calculate IQR and bounds
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Count values outside bounds
    Enum.count(values, fn v -> v < lower_bound || v > upper_bound end)
  end

  def count_outliers(_values), do: 0
end
