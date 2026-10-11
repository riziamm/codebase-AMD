function R = visit_check(newMat, oldMat)
%VISIT_CHECK  Which clinic visits are Repeat 1/2, how far apart they are, and which visit acuity belongs to.
%
%   R = visit_check('new.mat')                    % visits from the response-file names only
%   R = visit_check('new.mat', 'older.mat')       % + exam-date gaps from d.fDat in an older file
%
%   Prints AGGREGATES only: no participant codes, no dates of birth, no individual dates.
%   Copy the printed summary to the Paper 2 coder.

S = load(newMat, 'd');  d = S.d;
R = struct();

%% 1) Visits used per participant, from response-file names  (e.g. XX2_graph_LE_V2.png)
names = cellstr(d.filesRe(:));
tok   = regexp(names, '^(.*)_graph_(LE|RE)_V(\d+)', 'tokens', 'once');
ok    = ~cellfun(@isempty, tok);
code  = cellfun(@(t) t{1}, tok(ok), 'UniformOutput', false);
vis   = cellfun(@(t) str2double(t{3}), tok(ok));
ucode = unique(code, 'stable');
used  = cell(numel(ucode), 1);
for i = 1:numel(ucode)
    used{i} = unique(vis(strcmp(code, ucode{i})))';
end
pat = cellfun(@(v) ['V' strjoin(arrayfun(@num2str, v, 'UniformOutput', false), '+V')], used, 'UniformOutput', false);
[up, ~, ix] = unique(pat);  cnt = accumarray(ix, 1);
fprintf('\n1) Visits used per participant (Repeat 1 = first listed, Repeat 2 = last):\n');
for k = 1:numel(up), fprintf('     %-10s %2d participants\n', up{k}, cnt(k)); end
fprintf('   (%d of %d file names parsed)\n', sum(ok), numel(names));
R.visit_patterns = table(up, cnt, 'VariableNames', {'visits_used', 'n_participants'});

%% 2) Visit counts and last-visit years in d.T
vn  = d.T.VisitN;
yrs = cellfun(@(s) str2double(regexp(char(s), '\d{4}', 'match', 'once')), d.T.VisitLast);
fprintf('\n2) VisitN (visits per participant): '); fprintf('%d visits: %d   ', [unique(vn)'; histc(vn, unique(vn))']); fprintf('\n');
fprintf('   VisitLast years: '); fprintf('%d: %d   ', [unique(yrs(~isnan(yrs)))'; histc(yrs(~isnan(yrs)), unique(yrs(~isnan(yrs))))']); fprintf('\n');
fprintf('   Note: d.sevs holds ONE AREDS grade per eye (%s), so grade stability across visits cannot be checked from this file.\n', mat2str(size(d.sevs)));

%% 3) Exam-date gaps between the two visits used (needs d.fDat, e.g. in an older .mat)
if nargin < 2 || isempty(oldMat)
    fprintf('\n3) No older file given -> exam-date gaps not computed. Call visit_check(newMat, olderMatWith_fDat).\n');
    return
end
O = load(oldMat, 'd');  od = O.d;
if ~isfield(od, 'fDat')
    fprintf('\n3) %s has no d.fDat -> gaps unavailable. Ask Ted/Josh for exam dates.\n', oldMat);
    return
end
F = od.fDat;  if istable(F), F = table2cell(F); end
cols = od.fDatCols;  ci = @(n) find(strcmp(cols, n), 1);
need = {'SbjID', 'Visit', 'ExamDate'};
if any(cellfun(@(n) isempty(ci(n)), need))
    fprintf('\n3) d.fDat lacks one of SbjID / Visit / ExamDate -> gaps unavailable.\n'); return
end
sid = cellfun(@(x) char(string(x)), F(:, ci('SbjID')), 'UniformOutput', false);
v   = cellfun(@toNum,  F(:, ci('Visit')));
dtc = cellfun(@toDate, F(:, ci('ExamDate')), 'UniformOutput', false);  dt = vertcat(dtc{:});
fprintf('\n3) Exam dates parsed: %d of %d rows\n', sum(~isnat(dt)), numel(dt));

gap = nan(numel(ucode), 1);  lastYr = nan(numel(ucode), 1);
for i = 1:numel(ucode)
    m = strcmp(sid, ucode{i}) & ~isnat(dt);
    if ~any(m), continue, end
    vu = used{i};
    d1 = min(dt(m & v == vu(1)));  d2 = max(dt(m & v == vu(end)));
    if ~isempty(d1) && ~isempty(d2), gap(i) = days(d2 - d1); lastYr(i) = year(d2); end
end
g = gap(~isnan(gap));
if isempty(g)
    fprintf('   Participant codes in d.fDat did not match the file-name codes -> gaps unavailable.\n'); return
end
fprintf('   Repeat 1 -> Repeat 2 gap (days), %d participants: median %.0f, IQR %.0f-%.0f, range %.0f-%.0f\n', ...
    numel(g), median(g), prctile(g, 25), prctile(g, 75), min(g), max(g));
fprintf('   gap <= 30 d: %d   31-180 d: %d   181-365 d: %d   > 365 d: %d\n', ...
    sum(g <= 30), sum(g > 30 & g <= 180), sum(g > 180 & g <= 365), sum(g > 365));
R.gap_days_summary = [median(g) min(g) max(g)];

% last exam year vs d.T.VisitLast (same participant order assumed = d.T rows)
rn = d.T.Properties.RowNames;
if ~isempty(rn)
    [tf, loc] = ismember(ucode, rn);
    agree = sum(lastYr(tf) == yrs(loc(tf)));
    fprintf('   Year of the Repeat-2 exam matches VisitLast for %d of %d participants\n', agree, sum(tf & ~isnan(lastYr)));
    fprintf('   -> if (nearly) all match, acuity in d.T most likely belongs to the LAST visit (= Repeat 2). Confirm with Ted/Josh.\n');
end
end

% ---------------------------------------------------------------- helpers
function x = toNum(v)
if isnumeric(v), x = double(v(1)); else, x = str2double(regexp(char(string(v)), '\d+', 'match', 'once')); end
end

function t = toDate(v)
t = NaT;
try
    if isa(v, 'datetime'), t = v(1); return, end
    if isnumeric(v)
        if v > 7e5, t = datetime(v, 'ConvertFrom', 'datenum'); else, t = datetime(v, 'ConvertFrom', 'excel'); end
        return
    end
    s = strtrim(char(string(v)));
    for f = {'yyyy-MM-dd', 'dd/MM/yyyy', 'dd.MM.yyyy', 'dd-MMM-yyyy', 'yyyyMMdd', 'dd-MM-yyyy', 'MM/dd/yyyy'}
        try, t = datetime(s, 'InputFormat', f{1}); return, catch, end
    end
    t = datetime(s);
catch
    t = NaT;
end
end
