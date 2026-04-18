function export_cog_bci_behavior_tables(dataset_root, output_root)
% Export COG-BCI N-back behavioral MATLAB tables to CSV.
%
% Usage:
%   export_cog_bci_behavior_tables()
%   export_cog_bci_behavior_tables('/path/to/data/cog_bci')
%   export_cog_bci_behavior_tables('/path/to/data/cog_bci', '/path/to/out')
%
% Notes:
% - These files appear to store an MCOS MATLAB table named `nback`.
% - This script is expected to work in MATLAB.
% - Octave may fail to load these files because classdef/table support for
%   MCOS-serialized MAT files is incomplete. The script prints a per-file
%   status so you can immediately see whether your runtime can decode them.

  if nargin < 1 || isempty(dataset_root)
    dataset_root = fullfile(pwd, 'data', 'cog_bci');
  end
  if nargin < 2 || isempty(output_root)
    output_root = fullfile(dataset_root, 'behavior_csv');
  end

  try_load_octave_table_support();

  fprintf('Dataset root: %s\n', dataset_root);
  fprintf('Output root:  %s\n', output_root);

  if ~exist(dataset_root, 'dir')
    error('Dataset root does not exist: %s', dataset_root);
  end

  if ~exist(output_root, 'dir')
    mkdir(output_root);
  end

  patterns = {'0-Back.mat', '1-Back.mat', '2-Back.mat'};
  total_files = 0;
  exported_files = 0;
  failed_files = 0;

  subject_dirs = dir(fullfile(dataset_root, 'sub-*'));
  for i = 1:numel(subject_dirs)
    if ~subject_dirs(i).isdir
      continue;
    end

    subject_root = fullfile(subject_dirs(i).folder, subject_dirs(i).name);
    session_dirs = dir(fullfile(subject_root, 'ses-*'));

    for j = 1:numel(session_dirs)
      if ~session_dirs(j).isdir
        continue;
      end

      session_root = fullfile(session_dirs(j).folder, session_dirs(j).name);
      behavior_root = fullfile(session_root, 'behavioral');
      if ~exist(behavior_root, 'dir')
        continue;
      end

      for k = 1:numel(patterns)
        mat_path = fullfile(behavior_root, patterns{k});
        if ~exist(mat_path, 'file')
          continue;
        end

        total_files = total_files + 1;
        fprintf('\n[%d] Processing %s\n', total_files, mat_path);

        try
          payload = load(mat_path);
          table_name = resolve_nback_table_name(payload);
          if isempty(table_name)
            error('No nback table found in loaded variables. Loaded variables: %s', strjoin(fieldnames(payload), ', '));
          end

          nback = payload.(table_name);
          if ~istable(nback)
            error('Variable `%s` loaded, but it is not a table. Actual class: %s', table_name, class(nback));
          end

          relative_dir = strrep(behavior_root, dataset_root, '');
          if startsWith(relative_dir, filesep)
            relative_dir = relative_dir(2:end);
          end
          csv_dir = fullfile(output_root, relative_dir);
          if ~exist(csv_dir, 'dir')
            mkdir(csv_dir);
          end

          [~, stem, ~] = fileparts(mat_path);
          csv_path = fullfile(csv_dir, strcat(stem, '.csv'));
          writetable(nback, csv_path);

          fprintf('  table variable: %s\n', table_name);
          fprintf('  rows: %d\n', height(nback));
          fprintf('  columns: %d\n', width(nback));
          fprintf('  wrote: %s\n', csv_path);
          exported_files = exported_files + 1;
        catch err
          fprintf('  FAILED: %s\n', err.message);
          failed_files = failed_files + 1;
        end
      end
    end
  end

  fprintf('\nDone.\n');
  fprintf('  Total .mat files seen: %d\n', total_files);
  fprintf('  Exported CSV files:   %d\n', exported_files);
  fprintf('  Failed files:         %d\n', failed_files);
end

function try_load_octave_table_support()
  if exist('OCTAVE_VERSION', 'builtin') ~= 0
    try
      pkg('load', 'datatypes');
      fprintf('Loaded Octave package: datatypes\n');
    catch err
      fprintf('Could not load Octave package `datatypes`: %s\n', err.message);
    end
  end
end

function table_name = resolve_nback_table_name(payload)
  names = fieldnames(payload);
  table_name = '';

  for idx = 1:numel(names)
    name = names{idx};
    value = payload.(name);
    if istable(value)
      if strcmpi(name, 'nback')
        table_name = name;
        return;
      end
      if isempty(table_name)
        table_name = name;
      end
    end
  end
end
