function aircraft_parity(source_dir, output_dir)
% Synthetic-only parity probe for the unchanged author evaluation.m.
% source_dir contains the pinned original .m files, mounted read-only.
% No benchmark annotations or images are read. Run with GNU Octave CLI.
  addpath(source_dir);
  if ~exist(output_dir, 'dir'), mkdir(output_dir); end
  data_dir = fullfile(output_dir, 'synthetic-data');
  if ~exist(data_dir, 'dir'), mkdir(data_dir); end
  cases = {
    struct('name', 'perfect', 'truth', [1;2], 'x', [1;2], 'y', [1;2], 'scores', [1;1], 'expected', 1),
    struct('name', 'all_wrong', 'truth', [1;2], 'x', [1;2], 'y', [2;1], 'scores', [1;1], 'expected', 0),
    struct('name', 'imbalanced_classes', 'truth', [1;1;1;2], 'x', [1;2;3;4], 'y', [1;1;1;1], 'scores', [1;1;1;1], 'expected', .5),
    struct('name', 'missing_prediction', 'truth', [1;2], 'x', 1, 'y', 1, 'scores', 1, 'expected', .5),
    struct('name', 'no_predictions', 'truth', [1;2], 'x', zeros(0,1), 'y', zeros(0,1), 'scores', zeros(0,1), 'expected', 0),
    struct('name', 'finite_ties_sorted_class', 'truth', [1;2], 'x', [1;1;2;2], 'y', [2;1;2;1], 'scores', [1;1;1;1], 'expected', .5),
    struct('name', 'all_negative_infinity', 'truth', [1;2], 'x', [1;2], 'y', [1;2], 'scores', [-Inf;-Inf], 'expected', 0),
    struct('name', 'negative_finite_scores', 'truth', [1;2], 'x', [1;2], 'y', [1;2], 'scores', [-2;-1], 'expected', 1),
    struct('name', 'last_duplicate_pair_overwrites', 'truth', [1;2], 'x', [1;1;1;2], 'y', [1;1;2;2], 'scores', [2;0;1;1], 'expected', .5)
  };
  results = struct([]);
  for index = 1:numel(cases)
    item = cases{index};
    handle = fopen(fullfile(data_dir, 'images_variant_test.txt'), 'w');
    for image_index = 1:numel(item.truth)
      fprintf(handle, '%07d %s\n', image_index, char('A' + item.truth(image_index) - 1));
    end
    fclose(handle);
    record = struct('name', item.name, 'expected', item.expected, 'native_completed', false, ...
                    'mean_class_accuracy', [], 'confusion', [], 'error', '');
    try
      confusion = evaluation(data_dir, 'variant_test', item.x, item.y, item.scores);
      record.native_completed = true;
      record.mean_class_accuracy = mean(diag(confusion));
      record.confusion = confusion;
      if abs(record.mean_class_accuracy - item.expected) > 1e-12
        record.error = 'Official evaluator disagrees with prospective expectation';
      end
    catch exception
      record.error = exception.message;
    end
    results(index) = record;
  end
  report = struct('octave_version', version(), 'benchmark_images_read', 0, ...
                  'official_source_modified', false, 'synthetic_cases', results, ...
                  'all_cases_passed', all(arrayfun(@(r) r.native_completed && isempty(r.error), results)));
  handle = fopen(fullfile(output_dir, 'official-parity.json'), 'w');
  fprintf(handle, '%s\n', jsonencode(report));
  fclose(handle);
  if ~report.all_cases_passed
    error('Some official synthetic parity cases failed; inspect official-parity.json');
  end
end
