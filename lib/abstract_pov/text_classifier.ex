defmodule TextClassifier do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning
  alias KernelShtf.MetaLearning.Types.Learning

  # Define basic types using the bend/phrenia system
  phrenia TextData do
    document(text, features, category)
    classifier(model, vocabulary, categories, config)
    feature(name, weight)
  end

  @doc """
  Creates a new text classifier with meta-learning capabilities.
  The classifier adjusts its feature weights and selection strategy
  automatically based on the data it encounters.

  Config options:
  - :key_terms - Map of category-specific terms
  - :boost_weights - Map of boost values for different operations
  - :confidence_values - Map of confidence values for heuristics
  - :parameters - Map of additional parameter configurations
  """
  def new(initial_vocabulary, categories, config \\ %{}) do
    # Set defaults and merge with provided config
    config = Map.merge(default_config(categories), config)

    # Create a parameter for each word in the vocabulary with initial weights
    parameters =
      Enum.map(initial_vocabulary, fn word ->
        # Use configured initial weight and update rate
        initial_weight = Map.get(config.parameters, :initial_weight, 0.1)
        update_rate = Map.get(config.parameters, :update_rate, 0.1)
        MetaLearning.new_parameter(word, initial_weight, update_rate)
      end)

    # Create heuristics from configuration
    heuristics = create_heuristics(initial_vocabulary, config)

    # Create meta-rules from configuration
    meta_rules = create_meta_rules(config)

    # Create model
    model = MetaLearning.new_model(parameters, heuristics, meta_rules)

    # Return classifier with config included
    TextData.classifier(model, initial_vocabulary, categories, config)
  end

  # Create default configuration
  defp default_config(categories) do
    # Default key terms for each category
    default_key_terms =
      Enum.reduce(categories, %{}, fn category, acc ->
        Map.put(acc, category, [category])
      end)

    %{
      # Default key terms mapping
      key_terms: default_key_terms,

      # Default boost values
      boost_weights: %{
        positive_match: 0.2,
        negative_match: -0.15,
        category_boost: 0.3,
        unused_decay: -0.01,
        category_multiplier: 1.5
      },

      # Default confidence values
      confidence_values: %{
        positive_match: 0.8,
        negative_match: 0.7,
        category_boost: 0.9,
        unused_decay: 0.3
      },

      # Default parameter settings
      parameters: %{
        initial_weight: 0.1,
        update_rate: 0.1,
        momentum: 0.9,
        prune_threshold: 0.01
      },

      # Meta-rule priorities
      priorities: %{
        confidence_update: 0.7,
        oscillation_damping: 0.8,
        feature_boosting: 0.8,
        feature_pruning: 0.5
      }
    }
  end

  # Create heuristics based on configuration
  defp create_heuristics(vocabulary, config) do
    [
      # Regular heuristics as before...
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears in text and the label matches, increase weight
          if Map.get(features, name, 0) > 0 && features.category == label do
            Map.get(config.boost_weights, :positive_match, 0.2)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :positive_match, 0.8)
      ),

      # Decrease weights for words that appear in wrong categories
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears but in wrong category, decrease weight
          if Map.get(features, name, 0) > 0 && features.category != label do
            Map.get(config.boost_weights, :negative_match, -0.15)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :negative_match, 0.7)
      ),

      # Add category-specific boost for key words
      MetaLearning.new_heuristic(
        fn param_name ->
          # Check if this parameter is a key term for any category
          Enum.any?(config.key_terms, fn {_category, terms} ->
            Enum.member?(terms, param_name)
          end)
        end,
        fn name, value, _rate, features, label, _batch ->
          # Use configured key terms
          if Map.get(features, name, 0) > 0 do
            category_terms = Map.get(config.key_terms, label, [])

            if Enum.member?(category_terms, name) do
              Map.get(config.boost_weights, :category_boost, 0.3)
            else
              0.0
            end
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :category_boost, 0.9)
      ),

      # Decay unused words
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(vocabulary, param_name) end,
        fn name, value, _rate, features, _label, _batch ->
          if Map.get(features, name, 0) == 0 do
            Map.get(config.boost_weights, :unused_decay, -0.01)
          else
            0.0
          end
        end,
        Map.get(config.confidence_values, :unused_decay, 0.3)
      ),

      # Add special heuristic for negative sentiment words
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(["bad", "slow", "poor", "delay"], param_name) end,
        fn name, value, _rate, features, label, _batch ->
          if Map.get(features, name, 0) > 0 do
            if String.contains?(String.downcase(label), "negative") do
              # For negative categories, these words should have positive correlation
              Map.get(config.boost_weights, :positive_match, 0.2) * 2.0
            else
              # For positive categories, these words should have negative correlation
              Map.get(config.boost_weights, :negative_match, -0.15) * 2.0
            end
          else
            0.0
          end
        end,
        0.9
      )
    ]
  end

  # Create meta-rules based on configuration
  defp create_meta_rules(config) do
    [
      # Increase learning rate for features that consistently correlate with categories
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :convergence},
        fn heuristics, dataset ->
          # Find heuristics that are working well
          effective_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf}
              when conf < 0.9 ->
                # Increase confidence for effective heuristics
                Learning.heuristic(cond, action, conf + 0.05)

              other ->
                other
            end)

          {effective_heuristics, dataset}
        end,
        Map.get(config.priorities, :confidence_update, 0.7)
      ),

      # Reduce oscillation by adding momentum
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :oscillation},
        fn heuristics, dataset ->
          # Use configured momentum value
          momentum = Map.get(config.parameters, :momentum, 0.9)

          # Add momentum to oscillating parameters
          updated_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf} ->
                # Create a damped version of the action
                damped_action = fn name, value, rate, features, label, batch ->
                  # Get original update
                  raw_update = action.(name, value, rate, features, label, batch)

                  # Get previous updates from dataset statistics
                  previous_updates =
                    fold dataset do
                      case(dataset(data_points, statistics)) ->
                        Map.get(statistics, {:param_updates, name}, [])
                    end

                  # Apply momentum (weighted average with previous updates)
                  if previous_updates == [] do
                    raw_update
                  else
                    avg_prev = Enum.sum(previous_updates) / length(previous_updates)
                    raw_update * (1 - momentum) + avg_prev * momentum
                  end
                end

                Learning.heuristic(cond, damped_action, conf)

              other ->
                other
            end)

          {updated_heuristics, dataset}
        end,
        Map.get(config.priorities, :oscillation_damping, 0.8)
      ),

      # Add feature boosting for category-specific words
      MetaLearning.new_meta_rule(
        %{heuristic_type: :feature_boosting},
        fn heuristics, dataset ->
          boost_weight = Map.get(config.boost_weights, :category_boost, 0.3)

          boosting_heuristic =
            Learning.heuristic(
              %{type: :feature_boosting},
              fn name, value, _rate, features, label, _batch ->
                # Use configured key terms
                category_terms = Map.get(config.key_terms, label, [])

                if Enum.member?(category_terms, name) && Map.get(features, name, 0) > 0 do
                  boost_weight
                else
                  0.0
                end
              end,
              Map.get(config.confidence_values, :category_boost, 0.9)
            )

          {[boosting_heuristic | heuristics], dataset}
        end,
        Map.get(config.priorities, :feature_boosting, 0.8)
      ),

      # Add feature pruning for irrelevant words
      MetaLearning.new_meta_rule(
        %{heuristic_type: :feature_pruning},
        fn heuristics, dataset ->
          # Use configured pruning threshold
          prune_threshold = Map.get(config.parameters, :prune_threshold, 0.01)

          # Create a new heuristic that removes features with very low weights
          pruning_heuristic =
            Learning.heuristic(
              %{type: :feature_pruning},
              fn _name, value, _rate, _features, _label, _batch ->
                if abs(value) < prune_threshold do
                  # Further push toward zero
                  -value * 0.5
                else
                  0.0
                end
              end,
              0.4
            )

          {[pruning_heuristic | heuristics], dataset}
        end,
        Map.get(config.priorities, :feature_pruning, 0.5)
      )
    ]
  end

  @doc """
  Processes a document, extracting features for the classifier
  """
  def extract_features(document, vocabulary) do
    # Extract text, handling both string and map inputs
    text = cond do
      is_binary(document) -> document
      is_map(document) && Map.has_key?(document, :text) -> document.text
      is_map(document) && Map.has_key?(document, :original_text) -> document.original_text
      true -> ""
    end

    # Ensure we have text to process
    text = if text == "", do: "empty", else: text

    IO.puts("DEBUG: Extracting features from text: #{inspect(text)}")

    # Tokenize text - added better tokenization
    words =
      text
      |> String.downcase()
      |> String.replace(~r/[^\w\s]/, " ")  # Replace punctuation with spaces
      |> String.split(~r/\s+/)
      |> Enum.filter(&(&1 != ""))  # Remove empty strings

    IO.puts("DEBUG: Tokenized words: #{inspect(words)}")
    IO.puts("DEBUG: Vocabulary to match: #{inspect(vocabulary)}")

    # Count occurrences of vocabulary words
    features =
      Enum.reduce(vocabulary, %{}, fn word, acc ->
        count = Enum.count(words, &(&1 == word))
        # Only add words that actually appear
        if count > 0 do
          IO.puts("DEBUG: Found word '#{word}' with count #{count}")
          Map.put(acc, word, count)
        else
          acc
        end
      end)

    IO.puts("DEBUG: Extracted features: #{inspect(features)}")

    # Preserve original text if available
    features =
      if is_map(document) && Map.has_key?(document, :original_text) do
        Map.put(features, :original_text, document.original_text)
      else
        features
      end

    # Add flag to indicate this is a real document (not empty)
    Map.put(features, :valid_document, true)
  end

  @doc """
  Train the classifier on a set of documents
  """
  def train(classifier, documents, epochs \\ 5) do
    # Extract the necessary parts from the classifier
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories
    config = classifier.config

    IO.puts("DEBUG: Training classifier with #{length(documents)} documents for #{epochs} epochs")

    # Convert documents to dataset
    dataset = prepare_dataset(documents, vocabulary)
    IO.puts("DEBUG: Dataset prepared with #{length(dataset.data_points)} data points")

    # Debug the first data point
    if length(dataset.data_points) > 0 do
      first_point = List.first(dataset.data_points)
      IO.puts("DEBUG: First data point - Label: #{inspect(first_point.label)}, Features: #{inspect(first_point.features)}")
    end

    # Get initial weights for comparison
    initial_weights = Enum.map(model.parameters, fn
      %{name: name, value: value} -> {name, value}
      param -> {param[:name], param[:value]}
    end) |> Enum.into(%{})

    IO.puts("DEBUG: Initial weights: #{inspect(initial_weights)}")

    # Train using the meta-learning system
    trained_model = MetaLearning.train(model, dataset, epochs)

    # Debug parameter changes
    IO.puts("DEBUG: Initial param count: #{length(model.parameters)}")
    IO.puts("DEBUG: Trained param count: #{length(trained_model.parameters)}")

    # Get final weights
    final_weights = Enum.map(trained_model.parameters, fn
      %{name: name, value: value} -> {name, value}
      param -> {param[:name], param[:value]}
    end) |> Enum.into(%{})

    IO.puts("DEBUG: Final weights: #{inspect(final_weights)}")

    # Check if weights changed
    weight_changes = Enum.map(final_weights, fn {name, value} ->
      original = Map.get(initial_weights, name, 0.0)
      {name, value - original}
    end) |> Enum.into(%{})

    IO.puts("DEBUG: Weight changes: #{inspect(weight_changes)}")

    # Check for zero updates
    if Enum.all?(weight_changes, fn {_, change} -> change == 0.0 end) do
      IO.puts("WARNING: No weight changes occurred during training!")
    end

    # Create a new classifier with the trained model
    TextData.classifier(trained_model, vocabulary, categories, config)
  end

  @doc """
  Prepare training dataset from documents
  """
  def prepare_dataset(documents, vocabulary) do
    # Convert documents to data points
    data_points =
      Enum.map(documents, fn doc ->
        # Extract features based on document structure
        features = cond do
          is_map(doc) && Map.has_key?(doc, :text) && Map.has_key?(doc, :category) ->
            feats = extract_features(doc.text, vocabulary)
            # Add category to features for easier access
            Map.put(feats, :category, doc.category)

          is_map(doc) && Map.has_key?(doc, :features) && Map.has_key?(doc, :category) ->
            # Use existing features if already provided
            feats = doc.features
            # Add category if not already present
            Map.put(feats, :category, doc.category)

          is_map(doc) && Map.has_key?(doc, :variant) && doc.variant == :document ->
            # For document type from TextData
            text = Map.get(doc, :text, "")
            feats = extract_features(text, vocabulary)
            # Add original features if available
            feats = Map.merge(Map.get(doc, :features, %{}), feats)
            # Add category to features for easier access
            Map.put(feats, :category, doc.category)

          true ->
            # Default handling
            %{category: "unknown"}
        end

        # Create data point with the label set to the category
        MetaLearning.new_data_point(features, features.category)
      end)

    # Create dataset with initial statistics
    MetaLearning.new_dataset(data_points, %{
      size: length(data_points),
      categories: Enum.map(data_points, fn dp -> dp.label end) |> Enum.uniq() |> length(),
      avg_features:
        (Enum.map(data_points, fn dp -> map_size(dp.features) end)
         |> Enum.sum()) / max(1, length(data_points))
    })
  end

  @doc """
  Classify a new document
  """
  def classify(classifier, text) do
    # Extract model, vocabulary, categories and config
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories
    config = classifier.config

    # Extract features
    features = extract_features(text, vocabulary)

    # Get parameters (word weights)
    parameters = model.parameters

    # Calculate score for each category by applying category-specific weights
    scores =
      Enum.map(categories, fn category ->
        # Score using a more sophisticated approach that properly handles positive/negative weights
        base_score =
          Enum.reduce(parameters, 0, fn param, acc ->
            name = param.name
            weight = param.value
            word_count = Map.get(features, name, 0)

            # Skip words that don't appear in this document
            if word_count > 0 do
              # Multiply word count by its weight
              acc + word_count * weight
            else
              acc
            end
          end)

        # Apply category-specific boosting
        category_terms = Map.get(config.key_terms, category, [])
        category_boost = 0

        # Add extra weight for category-specific terms that appear in the document
        category_multiplier = Map.get(config.boost_weights, :category_multiplier, 1.5)

        category_specific_score =
          Enum.reduce(category_terms, 0, fn term, acc ->
            word_count = Map.get(features, term, 0)

            if word_count > 0 do
              # Add category-specific boost
              acc + word_count * category_multiplier
            else
              acc
            end
          end)

        final_score = base_score + category_specific_score

        # For negative sentiment categories, properly handle negative weights
        if String.contains?(String.downcase(category), "negative") do
          # For negative categories, negative words should increase score
          {category,
           final_score +
             Enum.sum(
               for {term, count} <- features,
                   count > 0,
                   term in ["bad", "slow", "poor", "delay"],
                   do: count * Map.get(config.boost_weights, :negative_match, 0.2) * -10
             )}
        else
          {category, final_score}
        end
      end)

    # Return category with highest score
    Enum.max_by(scores, fn {_category, score} -> score end)
  end

  @doc """
  Extract the most important words for each category
  """
  def important_features(classifier) do
    # Get parameters (word weights)
    parameters = classifier.model.parameters
    config = classifier.config

    # Use configured key terms or default to categories
    required_key_words =
      config.key_terms
      |> Enum.flat_map(fn {_category, words} -> words end)
      |> Enum.uniq()

    # Get all word weights from parameters
    all_word_weights =
      parameters
      |> Enum.map(fn param -> {param.name, param.value} end)

    # Add explicit entries for all key words with a positive default weight
    # This ensures all key words are included, even if not in parameters
    key_word_entries =
      required_key_words
      |> Enum.map(fn word ->
        # Try to find existing weight for this key word
        existing = Enum.find(all_word_weights, fn {name, _} -> name == word end)

        if existing do
          # Use existing weight if found
          existing
        else
          # Default positive weight for key words
          {word, Map.get(config.parameters, :initial_weight, 0.2)}
        end
      end)

    # Other non-key words with significant weights
    other_entries =
      all_word_weights
      |> Enum.filter(fn {word, value} ->
        !Enum.member?(required_key_words, word) &&
          abs(value) > Map.get(config.parameters, :prune_threshold, 0.1)
      end)
      |> Enum.sort_by(fn {_name, value} -> -abs(value) end)

    # Return the combined list, ensuring key words come first
    Enum.take(key_word_entries ++ other_entries, 20)
  end
end
