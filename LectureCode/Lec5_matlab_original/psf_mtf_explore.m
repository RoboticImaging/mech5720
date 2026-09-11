% PSF <-> MTF explorer: disc vs coded aperture
% Step through with Ctrl-Enter. Figure windows are reused throughout:
%   1  aperture, PSF and 2D MTF   (accumulates: disc on top, coded beneath)
%   2  1D MTF / noise-gain curves
%   3  image demo: sharp / blurred / spectra
%   4  deblurring results
%   5  Wiener: RMSE vs K, and the effective transfer
% Requires Image Processing Toolbox.

clearvars

% projector defaults: at 1280x720 every figure is downscaled, and MATLAB's
% default text and line weights disappear.
set(0, 'DefaultAxesFontSize',   14);
set(0, 'DefaultLineLineWidth',  2.0);
set(0, 'DefaultAxesLineWidth',  1.1);
set(0, 'DefaultTextFontSize',   14);

CMP = 'equalArea';   % 'equalArea'     : coded mask scaled to match the disc's open area
% 'equalDiameter' : coded mask fills the same envelope as the disc

D0 = 32;             % nominal defocus blur diameter, pixels
N  = 512;            % working image size
NoiseLevel = 1e-4;   % noise standard deviation, in units of the [0 1] signal

fprintf('\n=== setup: nominal blur diameter %d px, comparison mode %s ===\n', D0, CMP);

% 1. Disc aperture: PSF and 2D MTF
psfDisc = discPSF(D0);
showAperture(1, psfDisc, N, sprintf('disc, D = %d px', D0), 1);

% checkable prediction: first zero of the jinc sits at f = 1.2197 / D cyc/px
fNullPred = 1.2197 / D0;
[frTmp, ~, midTmp, ~] = bandMTF(psfDisc, N);
fNullMeas = firstNull(frTmp, midTmp);
fprintf('disc first null: predicted %.4f, measured %.4f cyc/px  (ratio %.3f)\n', ...
	fNullPred, fNullMeas, fNullMeas/fNullPred);

% 2. Disc MTF in 1D, with the nulls marked
[fr, loD, midD, hiD] = bandMTF(psfDisc, N);
showMTFband(2, {{fr, loD, midD, hiD}}, {'disc'}, [0 6*fNullPred 1], ...
	sprintf('Disc MTF, D = %d px  (band = min to max over orientation)', D0), ...
	fNullPred);

% 3. MTF scales inversely with the blur
Ds = [12 24 48];
sets = {}; labs = {};
for k = 1:numel(Ds)
	[frk, ~, mk, ~] = bandMTF(discPSF(Ds(k)), N);
	sets{k} = {frk, [], mk, []};
	labs{k} = sprintf('D = %d px', Ds(k));
	fprintf('D = %2d px : first null predicted %.4f, measured %.4f cyc/px\n', ...
		Ds(k), 1.2197/Ds(k), firstNull(frk, mk));
end
showMTFband(2, sets, labs, [0 0.30 1], ...
	'Disc MTF vs blur size: nulls move inward as the blur grows', []);

% 4. Apply the disc blur to a natural image
img = naturalImage(N);
showImageDemo(3, img, psfDisc, 'Natural image, disc blur', fNullMeas);

% 5. Apply it to a resolution target
tgt = resolutionTarget(N);
showImageDemo(3, tgt, psfDisc, 'Resolution target, disc blur', fNullMeas);
reportStarRings(N, D0);

% 6. Coded aperture: PSF and 2D MTF, added BENEATH the disc in window 1
[psfCoded, Dcoded] = codedPSF(D0, CMP);
showAperture(1, psfCoded, N, sprintf('coded, envelope %d px (%s)', Dcoded, CMP), 2);

% 7. Disc vs coded, 1D, with the orientation band shown
[frC, loC, midC, hiC] = bandMTF(psfCoded, N);
XMAX = min(0.5, 6*fNullMeas);

showMTFband(2, {{fr, loD, midD, hiD}, {frC, loC, midC, hiC}}, {'disc','coded'}, ...
	[0 XMAX 1], 'Disc vs coded: shaded band is min to max over orientation', ...
	fNullMeas);

% 8. The coded aperture on the same two images
showImageDemo(3, img, psfCoded, 'Natural image, coded blur', []);

showImageDemo(3, tgt, psfCoded, 'Resolution target, coded blur', []);

% 9. Deblurring, naive: Y = H X, so try X_hat = Y / H
showNaive(4, img, psfDisc, 0, 'disc, no noise', fNullMeas);

% 10. The same thing with a very small amount of noise
showNaive(4, img, psfDisc, NoiseLevel, sprintf('disc, noise sigma = %g', NoiseLevel), fNullMeas);

% 11. And on the resolution target, where you can see WHICH rings die
showNaive(4, tgt, psfDisc, NoiseLevel, sprintf('target, noise sigma = %g', NoiseLevel), fNullMeas);

% 12. The same noise, through the coded aperture
showNaive(4, img, psfCoded, NoiseLevel, sprintf('coded, noise sigma = %g', NoiseLevel), []);

% 13. Why: the noise gain 1/|H| against frequency
compareGain(2, {psfDisc, psfCoded}, {'disc','coded'}, N, XMAX, fNullMeas);

% 15. Wiener: same noise, same PSF, but stop dividing by tiny numbers
K0 = 2e-7;                                   % a reasonable starting point
showWienerVsNaive(4, img, psfDisc, NoiseLevel, K0, 'natural image, disc');

% 16. The same on the resolution target
showWienerVsNaive(4, tgt, psfDisc, NoiseLevel, K0, 'resolution target, disc');

% 17. Sweep K and watch the error turn around
Ks = logspace(-11, -1, 31);
[KbestD, rD] = sweepK(5, tgt, psfDisc, NoiseLevel, Ks, 'disc');
fprintf('disc : best K = %.2e, RMSE %.4f\n', KbestD, min(rD));

% 18. Three values of K, side by side
showKtriple(4, tgt, psfDisc, NoiseLevel, [KbestD/3000, KbestD, KbestD*3000]);

% 20. The coded aperture under Wiener
[KbestC, rC] = sweepK(5, tgt, psfCoded, NoiseLevel, Ks, 'coded');
fprintf('coded: best K = %.2e, RMSE %.4f   (disc %.4f, so %.0f%% better)\n', ...
	KbestC, min(rC), min(rD), 100*(min(rD)-min(rC))/min(rD));

% 20b. Disc vs coded, both Wiener-deblurred, side by side
showWienerPair(4, tgt, psfDisc, psfCoded, NoiseLevel, KbestD, KbestC, 'disc', 'coded');

% 20c. The same on the natural image
showWienerPair(4, img, psfDisc, psfCoded, NoiseLevel, KbestD, KbestC, 'disc', 'coded');

% 21. Both curves together
showTwoSweeps(5, Ks, rD, rC, {'disc','coded'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % 22. Back to the spatial domain: w = ifft2(W)
% showSpatialKernel(4, psfCoded, 1e-2, N);
% 
% % 23. Deconvolution by convolution: the same filter, done the slow way
% % Take the full kernel w = ifft2(W) and convolve with it circularly. No
% % truncation, so the result is not an approximation of the FFT answer, it IS
% % the FFT answer: the two agree to machine precision. Done on a small patch,
% % because the full kernel is as big as the image.
% % showSpatialExact(4, tgt, psfCoded, NoiseLevel, 1e-3, 129);
% showSpatialExact(4, img, codedPSF(13, CMP), NoiseLevel, 1e-3, 129)
% 
% % 24. What it costs
% % Spatial convolution is N^2*M^2; the FFT route is about 2*N^2*log2(N^2). With
% % a full-size kernel M = N, so the gap grows as N^2/log2(N^2). This is why
% % nobody deconvolves a whole image in the spatial domain. It earns its place
% % when the PSF varies across the field, when the prior is not Gaussian (Levin's
% % sparse-prior solver runs in the spatial domain for exactly that reason), or
% % in hardware with no frame buffer.
% reportSpatialCost([129 512 1024 4096]);



% ======================================================================
% local functions: Wiener section
% ======================================================================

function [y, xhat, H] = wienerCycle(x, psf, sigma, K, seed)
% Blur, add noise, then Wiener filter. Symmetric padding so the DFT's
% periodicity does not contaminate the edges. K = [] gives the naive inverse.
if nargin < 5, seed = 0; end
pad = size(psf,1);
xp  = padarray(x, [pad pad], 'symmetric', 'both');
Np  = size(xp,1);
H   = otf(psf, Np);
rng(seed);
yp = real(ifft2(fft2(xp) .* H));
if sigma > 0, yp = yp + sigma*randn(size(yp)); end
if isempty(K)
	Wf = 1 ./ H;
else
	Wf = conj(H) ./ (abs(H).^2 + K);
end
xhp  = real(ifft2(Wf .* fft2(yp)));
y    = yp(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
xhat = xhp(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
end

function showWienerVsNaive(fignum, x, psf, sigma, K, ttl)
% Window 4: blurred capture, naive estimate, Wiener estimate, and the error.
[y, xn] = wienerCycle(x, psf, sigma, []);
[~, xw] = wienerCycle(x, psf, sigma, K);
rn = sqrt(mean((xn(:)-x(:)).^2));
rw = sqrt(mean((xw(:)-x(:)).^2));
fprintf('%-32s naive RMSE %.4g , Wiener RMSE %.4f  (K = %.1e)\n', ttl, rn, rw, K);

sfigure(fignum); clf;
ddsubplot(2,2,1); imagesc(y);  axis image off; colormap(gca,'gray'); caxis([0 1]);
title('blurred + noise');
ddsubplot(2,2,2); imagesc(xn); axis image off; colormap(gca,'gray');
caxis(wRobustLim(xn)); title(sprintf('naive 1/H   RMSE %.3g', rn));
ddsubplot(2,2,3); imagesc(xw); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('Wiener, K = %.1e   RMSE %.3f', K, rw));
ddsubplot(2,2,4); imagesc(abs(xw-x)); axis image off; colormap(gca,'gray');
caxis([0 0.3]); title(sprintf('|Wiener error|   (%s)', ttl), 'Interpreter','none');
drawnow;
end

function showWienerPair(fignum, x, psfA, psfB, sigma, KA, KB, labA, labB)
% Window 4: the same scene, same noise, both apertures, each Wiener-deblurred
% at its own best K. Two large panels, for the projector.
[~, xa] = wienerCycle(x, psfA, sigma, KA);
[~, xb] = wienerCycle(x, psfB, sigma, KB);
ra = sqrt(mean((xa(:)-x(:)).^2));
rb = sqrt(mean((xb(:)-x(:)).^2));
fprintf('Wiener, %s RMSE %.4f (K=%.1e) vs %s RMSE %.4f (K=%.1e)  -> %.0f%% better\n', ...
	labA, ra, KA, labB, rb, KB, 100*(ra-rb)/ra);

sfigure(fignum); clf;
ddsubplot(1,2,1);
imagesc(xa); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('%s,  RMSE %.3f', labA, ra), 'FontSize', 16);
ddsubplot(1,2,2);
imagesc(xb); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('%s,  RMSE %.3f', labB, rb), 'FontSize', 16);
drawnow;
end

function [Kbest, rmse] = sweepK(fignum, x, psf, sigma, Ks, lab)
% Window 5: RMSE against K, with the minimum marked.
rmse = zeros(size(Ks));
for i = 1:numel(Ks)
	[~, xh] = wienerCycle(x, psf, sigma, Ks(i));
	rmse(i) = sqrt(mean((xh(:)-x(:)).^2));
end
[~, i] = min(rmse);  Kbest = Ks(i);
sfigure(fignum); clf;
loglog(Ks, rmse, 'LineWidth', 2); hold on; grid on;
loglog(Kbest, rmse(i), 'o', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('K'); ylabel('RMSE of the estimate');
title(sprintf('%s: RMSE vs K, best K = %.1e', lab, Kbest));
legend({lab, 'minimum'}, 'Location','north'); hold off; drawnow;
end

function showTwoSweeps(fignum, Ks, r1, r2, labs)
% Window 5: two RMSE-vs-K curves on the same axes.
sfigure(fignum); clf;
loglog(Ks, r1, 'LineWidth', 2); hold on; grid on;
loglog(Ks, r2, 'LineWidth', 2);
xlabel('K'); ylabel('RMSE of the estimate');
title('Wiener: coded vs disc, across K');
legend(labs, 'Location','north'); hold off; drawnow;
end

function showKtriple(fignum, x, psf, sigma, Ks)
% Window 4: three values of K side by side.
sfigure(fignum); clf;
names = {'too small', 'about right', 'too large'};
for i = 1:3
	[~, xh] = wienerCycle(x, psf, sigma, Ks(i));
	r = sqrt(mean((xh(:)-x(:)).^2));
	ddsubplot(1,3,i);
	imagesc(xh); axis image off; colormap(gca,'gray'); caxis([0 1]);
	title(sprintf('%s\nK = %.0e,  RMSE %.3f', names{i}, Ks(i), r));
end
drawnow;
end

function showEffective(fignum, psf, N, Ks, fNull, lab)
% Window 5: the effective transfer |W H| at several K.
% Uses the WORST orientation at each radius: a radial mean blends the disc's
% zeros with neighbouring non-zero radii and hides exactly what we are after.
H = otf(psf, N);
c = floor(N/2) + 1;
[xx,yy] = meshgrid((1:N)-c, (1:N)-c);
rr = sqrt(xx.^2 + yy.^2);
nb = floor(N/2);
sfigure(fignum); clf; hold on; grid on;
labs = cell(1,numel(Ks));
for j = 1:numel(Ks)
	E = abs( (conj(H)./(abs(H).^2 + Ks(j))) .* H );
	Es = fftshift(E);
	prof = zeros(1,nb); fr = zeros(1,nb);
	for k = 1:nb
		sel = rr >= k-1 & rr < k;
		prof(k) = min(Es(sel));  fr(k) = (k-0.5)/N;
	end
	semilogy(fr, max(prof,1e-6), 'LineWidth', 1.8);
	labs{j} = sprintf('K = %.0e', Ks(j));
end
if ~isempty(fNull) && ~isnan(fNull)
	yl = ylim; plot([fNull fNull], yl, 'k--', 'HandleVisibility','off');
end
if isempty(fNull) || isnan(fNull)
	fxmax = 0.25;
else
	fxmax = min(0.5, 8*fNull);
end
set(gca,'YScale','log'); xlim([0 fxmax]); ylim([1e-4 2]);
xlabel('spatial frequency, cycles/pixel');
ylabel('|W H|, worst orientation');
title(sprintf('%s: what survives the round trip. No K rescues a true null.', lab));
legend(labs, 'Location','southwest'); hold off; drawnow;
end

function lim = wRobustLim(a)
% Percentile display limits, so one huge outlier does not flatten the image.
v = sort(a(:));
lo = v(max(1,round(0.02*numel(v))));
hi = v(min(numel(v),round(0.98*numel(v))));
if hi <= lo, hi = lo + eps; end
lim = [lo hi];
end


% ======================================================================
% local functions: naive-deblur section
% ======================================================================

function showNaive(fignum, x, psf, sigma, ttl, fNull)
% Window 4: blur, add noise, then divide by H and look at the wreckage.
pad = size(psf,1);
xp  = padarray(x, [pad pad], 'symmetric', 'both');
Np  = size(xp,1);
H   = otf(psf, Np);

yp = real(ifft2(fft2(xp) .* H));
if sigma > 0
	rng(0);                                  % same noise every run
	yp = yp + sigma*randn(size(yp));
end
xhp = real(ifft2(fft2(yp) ./ H));            % the naive inverse filter

cropf = @(a) a(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
y    = cropf(yp);
xhat = cropf(xhp);

rmse = sqrt(mean((xhat(:)-x(:)).^2));
fprintf('%-34s : peak gain 1/min|H| = %.2e , RMSE of estimate = %.3g\n', ...
	ttl, 1/min(abs(H(:))), rmse);

sfigure(fignum); clf;
ddsubplot(2,2,1);
imagesc(y); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('blurred%s', tern(sigma>0, ' + noise', '')));

ddsubplot(2,2,2);
imagesc(xhat); axis image off; colormap(gca,'gray'); caxis(robustLim(xhat));
title(sprintf('naive X_{hat} = Y / H   (%s)', ttl), 'Interpreter','tex');

ddsubplot(2,2,3);
imagesc(xhat); axis image off; colormap(gca,'gray'); caxis([0 1]);
title('same estimate, clipped to [0 1]');

ddsubplot(2,2,4);
[fr, lo, mid, hi] = bandMTF(psf, size(x,1));
semilogy(fr, 1./max(mid,1e-12), 'LineWidth', 1.6); hold on; grid on;
semilogy(fr, 1./max(lo, 1e-12), ':', 'LineWidth', 1.2);
if ~isempty(fNull) && ~isnan(fNull)
	yl = ylim; plot([fNull fNull], yl, 'k--');
end

if isempty(fNull) || isnan(fNull)
	fxmax = 0.25;                       % no null supplied: show a sensible band
else
	fxmax = min(0.5, 8*fNull);
end
xlim([0 fxmax]);

xlabel('cycles/pixel'); ylabel('noise gain 1/|H|');
legend({'mean','worst orientation'}, 'Location','northwest');
title('what the inverse filter multiplies the noise by'); hold off;
drawnow;
end

function compareGain(fignum, psfs, labs, N, xmax, fNull)
% Window 2: noise gain 1/|H| for several apertures on one pair of axes.
cols = lines(numel(psfs));
sfigure(fignum); clf;
ddsubplot(2,1,1); hold on; grid on;
for k = 1:numel(psfs)
	[fr, ~, mid, ~] = bandMTF(psfs{k}, N);
	plot(fr, max(mid,1e-12), 'LineWidth', 1.8, 'Color', cols(k,:));
end
set(gca,'YScale','log'); ylim([1e-5 1.5]); xlim([0 xmax]);
ylabel('MTF'); title('MTF, and the noise gain it implies');
legend(labs,'Location','northeast'); hold off;

ddsubplot(2,1,2); hold on; grid on;
for k = 1:numel(psfs)
	[fr, lo, ~, ~] = bandMTF(psfs{k}, N);
	plot(fr, 1./max(lo,1e-12), 'LineWidth', 1.8, 'Color', cols(k,:));
end
if ~isempty(fNull), yl = ylim; plot([fNull fNull], yl, 'k--', 'HandleVisibility','off'); end
set(gca,'YScale','log'); xlim([0 xmax]);
xlabel('spatial frequency, cycles/pixel');
ylabel('1/|H|, worst orientation');
legend(labs,'Location','northwest'); hold off;
drawnow;
end

function lim = robustLim(a)
% Display limits from percentiles, so one enormous outlier does not flatten
% the whole image to grey. Avoids prctile (Statistics Toolbox).
v = sort(a(:));
lo = v(max(1,round(0.02*numel(v))));
hi = v(min(numel(v),round(0.98*numel(v))));
if hi <= lo, hi = lo + eps; end
lim = [lo hi];
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end

function rmse = bandLimitedInverse(x, psf, sigma, fcut) %#ok<DEFNU>
% Invert only out to fcut and discard the rest. The crude precursor to Wiener.
% Not called by any cell above; kept for the "what actually helps" discussion.
pad = size(psf,1);
xp  = padarray(x, [pad pad], 'symmetric', 'both');
Np  = size(xp,1);
H   = otf(psf, Np);
rng(0);
yp  = real(ifft2(fft2(xp) .* H)) + sigma*randn(Np);
fv  = [0:floor(Np/2) -(ceil(Np/2)-1):-1] / Np;
[FX,FY] = meshgrid(fv, fv);
keep = hypot(FX,FY) <= fcut;
xhp = real(ifft2( (fft2(yp) ./ H) .* keep ));
xh  = xhp(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
rmse = sqrt(mean((xh(:)-x(:)).^2));
end


% ======================================================================
% local functions: core
% ======================================================================

function p = discPSF(D)
% Unit-sum disc of diameter D pixels, on an odd-sized support.
r = ceil(D/2) + 1;  n = 2*r + 1;
[xx,yy] = meshgrid(-r:r, -r:r);
p = double( (xx.^2 + yy.^2) <= (D/2)^2 );
p = p / sum(p(:));
end

function [p, Denv] = codedPSF(Dref, mode)
% Zhou & Nayar sigma = 0.001, 13x13. Scaled either to the same envelope as the
% disc, or to the same open AREA (equal photons, the fair-light comparison).
M = zhouNayarMask();
tau = mean(M(:));                 % open fraction inside the square
discArea = pi*(Dref/2)^2;
switch lower(mode)
	case 'equaldiameter'
		Denv = Dref;
	case 'equalarea'
		% side s of the coded square such that tau*s^2 = disc open area
		Denv = round(sqrt(discArea / tau));
	otherwise
		error('mode must be equalArea or equalDiameter');
end
Denv = max(Denv, 13);                          % keep at least 1 px per cell
p = imresize(M, [Denv Denv], 'nearest');
p = double(p > 0.5);
if mod(size(p,1),2)==0, p = padarray(p,[1 1],0,'post'); end
p = p / sum(p(:));
end

function M = zhouNayarMask()
% 13x13, 1 = open. Transcribed from Zhou & Nayar ICCP 2009 Fig 3 and checked
% against their reported photometry (44% of a circular aperture's throughput).
rows = { '0000000000000', ...
	'0001011111000', ...
	'0000000110000', ...
	'0101000111110', ...
	'0100111100010', ...
	'0110000001010', ...
	'0110111001000', ...
	'0110100011010', ...
	'0110100000010', ...
	'0110010100110', ...
	'0011100111100', ...
	'0000000111000', ...
	'0000000000000' };
M = zeros(13,13);
for i = 1:13, M(i,:) = double(rows{i}) - double('0'); end
end

function a = openArea(p) %#ok<DEFNU>
% Open area in pixels (the PSF is unit-sum, so count its support).
a = sum(p(:) > 0);
end

function showAperture(fignum, psf, N, ttl, row)
% Window 1, ACCUMULATING: row 1 is the disc, row 2 the coded mask, so both are
% on screen together. Only row 1 clears the figure.
if nargin < 5, row = 1; end
H = otf(psf, N);
sfigure(fignum);
if row == 1, clf; end
ddsubplot(2,2,(row-1)*2 + 1);
imagesc(psf); axis image off; colormap(gca,'gray');
title(sprintf('PSF: %s', ttl), 'Interpreter','none');
ddsubplot(2,2,(row-1)*2 + 2);
imagesc(20*log10(fftshift(abs(H)) + 1e-6)); axis image off;
colormap(gca,'parula'); caxis([-60 0]); colorbar;
title('MTF, dB (2D)');
drawnow;
end

function [fr, lo, mid, hi] = bandMTF(psf, N)
% For each radius: the MIN, MEAN and MAX of the MTF over all orientations.
% The disc is rotationally symmetric so its band collapses to a line, and its
% nulls are whole RINGS. The coded mask's weak spots are isolated in angle, so
% its band stays open even where its minimum dips.
H = fftshift(abs(otf(psf, N)));
c = floor(N/2) + 1;
[xx,yy] = meshgrid((1:N)-c, (1:N)-c);
rr = sqrt(xx.^2 + yy.^2);
nb = floor(N/2);
lo = zeros(1,nb); mid = lo; hi = lo; fr = lo;
for k = 1:nb
	sel = rr >= k-1 & rr < k;
	v = H(sel);
	lo(k) = min(v); mid(k) = mean(v); hi(k) = max(v);
	fr(k) = (k-0.5) / N;
end
d = mid(1);
lo = lo/d; mid = mid/d; hi = hi/d;
end

function showMTFband(fignum, sets, labs, xl, ttl, fMark)
% Window 2: mean MTF as a line, with min-to-max over orientation shaded behind.
% sets{k} = {fr, lo, mid, hi};  pass lo = hi = [] for a plain line.
cols = lines(numel(sets));
sfigure(fignum);
clf;
for pass = 1:2
	ddsubplot(2,1,pass);
	hold on; grid on;
	for k = 1:numel(sets)
		fr = sets{k}{1}; lo = sets{k}{2}; mid = sets{k}{3}; hi = sets{k}{4};
		m = fr <= xl(2);
		if ~isempty(lo)
			fill([fr(m) fliplr(fr(m))], ...
				[max(hi(m),1e-6) fliplr(max(lo(m),1e-6))], cols(k,:), ...
				'FaceAlpha',0.20, 'EdgeColor','none', 'HandleVisibility','off');
		end
		plot(fr(m), max(mid(m),1e-6), 'LineWidth', 1.8, 'Color', cols(k,:));
	end
	if ~isempty(fMark)
		yl = ylim; plot([fMark fMark], yl, 'k--', 'HandleVisibility','off');
	end
	xlim([xl(1) xl(2)]); xlabel('spatial frequency, cycles/pixel');
	if pass == 2
		set(gca,'YScale','log'); ylim([1e-5 1.5]); ylabel('MTF (log)');
	else
		ylim([0 xl(3)]); ylabel('MTF'); title(ttl);
	end
	legend(labs, 'Location','northeast'); hold off;
end
drawnow;
end

function H = otf(psf, N)
% OTF on an N x N grid, normalised so H(0) = 1 (the PSF already sums to 1).
P = zeros(N);
s = size(psf,1);
o = floor(N/2) - floor(s/2) + 1;
P(o:o+s-1, o:o+s-1) = psf;
P = ifftshift(P);
H = fft2(P);
end

function f0 = firstNull(fr, m)
% Frequency of the first local minimum that is essentially zero.
f0 = NaN;
for k = 2:numel(m)-1
	if m(k) < m(k-1) && m(k) < m(k+1) && m(k) < 0.05
		f0 = fr(k); return;
	end
end
end

function y = blurFFT(x, psf)
% Blur in the frequency domain, with symmetric padding so the DFT's circular
% wrap-around does not contaminate the edges.
s = size(psf,1);  pad = s;
xp = padarray(x, [pad pad], 'symmetric', 'both');
N  = size(xp,1);
H  = otf(psf, N);
yp = real(ifft2(fft2(xp) .* H));
y  = yp(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
end

function showImageDemo(fignum, x, psf, ttl, fNull)
% Window 3: sharp and blurred above, their log spectra below.
y = blurFFT(x, psf);
sfigure(fignum); clf;
ddsubplot(2,2,1); imagesc(x); axis image off; colormap(gca,'gray'); caxis([0 1]);
title('sharp');
ddsubplot(2,2,2); imagesc(y); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(ttl, 'Interpreter','none');
ddsubplot(2,2,3); showSpec(x, fNull); title('spectrum, sharp');
ddsubplot(2,2,4); showSpec(y, fNull); title('spectrum, blurred');
drawnow;
end

function showSpec(x, fNull)
% Log-magnitude spectrum, with the first null radius drawn on if supplied.
% NOTE: this circle is in the FREQUENCY plane. The grey ring you see in the
% blurred star is at a different, SPATIAL radius: r = NCYC/(2*pi*f_null).
N = size(x,1);
S = fftshift(abs(fft2(x)));
S = S - min(S(:));
S = S ./ max(S(:));
imagesc(20*log10(S + 1e-6));
% imagesc(S.^0.125);
axis image off; colormap(gca,'parula');
if ~isempty(fNull) && ~isnan(fNull)
	c = floor(N/2)+1;  r = fNull * N;
	th = linspace(0,2*pi,256);
	hold on; plot(c + r*cos(th), c + r*sin(th), 'r--', 'LineWidth', 1.2); hold off;
end
end

function img = naturalImage(N)
try
	a = im2double(imread('cameraman.tif'));
catch
	a = im2double(rgb2gray(imread('peppers.png')));
end
img = imresize(a, [N N]);
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
end

function t = resolutionTarget(N)
% Siemens star. Local frequency is NCYC/(2*pi*r), so it falls smoothly with
% radius: an MTF null becomes a grey RING at a predictable radius, and the
% contrast reversal past each null shows as the spokes swapping polarity.
NCYC = starCycles();
[xx, yy] = meshgrid((1:N) - N/2, (1:N) - N/2);
th = atan2(yy, xx);
rr = hypot(xx, yy);
t = 0.5 + 0.5*sign(sin(NCYC*th));
rAlias = NCYC/(2*pi*0.5);          % inside this radius the star aliases
t(rr < max(rAlias, 14)) = 0.5;     % blank the core
t(rr > N/2 - 6)         = 0.5;     % and the corners
t = min(max(t, 0), 1);
end

function n = starCycles()
% Cycles around the star. Fewer = chunkier spokes, which projects better.
n = 36;
end

function reportStarRings(N, D) %#ok<INUSL>
% Where the grey rings should appear in the blurred star, from the jinc zeros.
J1 = [3.8317 7.0156 10.1735 13.3237 16.4706];
fn = J1 / (pi*D);
r  = starCycles() ./ (2*pi*fn);
fprintf('star rings (D = %d px, %d cycles):\n', D, starCycles());
for k = 1:numel(fn)
	fprintf('   null f = %.4f cyc/px  ->  grey ring at r = %.0f px%s\n', ...
		fn(k), r(k), tern(r(k) < N/2-6, '', '   (outside the frame)'));
end
end


% ======================================================================
% local functions: spatial-domain deconvolution
% ======================================================================

function n = oddSize(n)
% Force an ODD transform size. With an even N, fftshift centres the kernel on a
% half-pixel: it is then symmetric about no index at all, and any odd crop is
% off-centre by half a pixel. For a sharpening kernel that is fatal.
if mod(n,2)==0, n = n + 1; end
end


function w = wienerKernel(psf, K, Np)
% The Wiener filter as a spatial convolution kernel, centred.
H = otf(psf, Np);
W = conj(H) ./ (abs(H).^2 + K);
w = real(fftshift(ifft2(W)));
end

function k = cropKernel(w, hw, mode)
% Crop w to an ODD (2*hw+1)^2 kernel centred on its peak.
%   mode 1  raw crop
%   mode 2  crop, then renormalise so the DC gain matches the full kernel
%   mode 3  crop, raised-cosine window, then renormalise
% Odd size matters: conv2(...,'same') centres an odd kernel exactly, and an
% even one lands half a pixel off.
if nargin < 3, mode = 3; end
c = floor(size(w,1)/2) + 1;
S = sum(w(:));
k = w(c-hw:c+hw, c-hw:c+hw);
if mode >= 3
	[a,b] = meshgrid(-hw:hw, -hw:hw);
	rr = min(hypot(a,b)/hw, 1);
	k = k .* (0.5*(1 + cos(pi*rr)));      % raised cosine, zero at the edge
end
if mode >= 2
	k = k * (S / sum(k(:)));
end
end

function showSpatialKernel(fignum, psf, K, N)
% Window 4: the kernel, a slice through it, and a check that its DFT is W.
pad = size(psf,1);  Np = oddSize(N + 2*pad);
w = wienerKernel(psf, K, Np);
c = floor(Np/2) + 1;  hw = 40;
patch = w(c-hw:c+hw, c-hw:c+hw);

fprintf('kernel: centre %+.3f, min %+.3f, max %+.3f, sum %.4f\n', ...
	w(c,c), min(w(:)), max(w(:)), sum(w(:)));

sfigure(fignum); clf;
ddsubplot(2,2,1);
imagesc(patch); axis image off; colormap(gca,'gray'); caxis(wRobustLim(patch));
title(sprintf('w = ifft2(W), central %dx%d', 2*hw+1, 2*hw+1));

ddsubplot(2,2,2);
plot(-hw:hw, w(c, c-hw:c+hw), 'LineWidth', 1.8); grid on;
xlabel('pixels from centre'); ylabel('w');
title('slice: positive centre, negative surround = sharpening');

ddsubplot(2,2,3);
semilogy(-hw:hw, max(abs(w(c, c-hw:c+hw)),1e-8), 'LineWidth', 1.6); grid on;
xlabel('pixels from centre'); ylabel('|w|, log');
title('how fast it decays: this is what sets the kernel size');

ddsubplot(2,2,4);
Wback = abs(fftshift(fft2(ifftshift(w))));
plot(linspace(-0.5,0.5,Np), Wback(c,:), 'LineWidth', 1.6); grid on;
xlim([-0.5 0.5]); xlabel('cycles/pixel'); ylabel('|W|');
title('transform of w: the Wiener filter, as it should be');
drawnow;
end

function showSpatialDeconv(fignum, x, psf, sigma, K, hws, lab)
% Window 4: the exact FFT result against convolution with a cropped kernel.
% No windowing and no renormalising: a plain crop is the honest test of whether
% the kernel is compact. (Renormalising by a scale factor is actively bad here,
% because a sharpening kernel's partial sums swing through zero.)

pad = max(size(psf,1), max(hws));
xp  = padarray(x, [pad pad], 'symmetric', 'both');

if mod(size(xp,1),2)==0
	xp = padarray(xp, [1 1], 'symmetric', 'post');   % odd grid, see oddSize
end
Np  = size(xp,1);
H   = otf(psf, Np);
rng(0);
yp = real(ifft2(fft2(xp) .* H)) + sigma*randn(Np);
Wf = conj(H) ./ (abs(H).^2 + K);
exact = real(ifft2(Wf .* fft2(yp)));
exact = exact(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
rex = sqrt(mean((exact(:)-x(:)).^2));
w = real(fftshift(ifft2(Wf)));
c = floor(Np/2) + 1;

fprintf('\n%s, K = %.0e : exact FFT RMSE vs sharp = %.4f\n', lab, K, rex);
fprintf('   %9s %10s %12s\n', 'kernel', 'RMSE', 'vs exact');
best = inf; bestk = []; bestSize = 0;
for i = 1:numel(hws)
	hw = hws(i);
	k  = w(c-hw:c+hw, c-hw:c+hw);
	ap = conv2(yp, k, 'same');
	ap = ap(pad+1:pad+size(x,1), pad+1:pad+size(x,2));
	r  = sqrt(mean((ap(:)-x(:)).^2));
	fprintf('   %5dx%-3d %10.4f %11.1fx\n', 2*hw+1, 2*hw+1, r, r/rex);
	if r < best, best = r; bestk = ap; bestSize = 2*hw+1; end
end

sfigure(fignum); clf;
ddsubplot(1,2,1); imagesc(exact); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('%s: exact, via the FFT   RMSE %.3f', lab, rex), 'FontSize', 14);
ddsubplot(1,2,2); imagesc(bestk); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('best kernel %dx%d   RMSE %.3f', bestSize, bestSize, best), 'FontSize', 14);
drawnow;
end

function reportCompactness(psf, N, Ks)
% How much of the kernel's energy sits within a given radius. A kernel you can
% crop is one whose energy is concentrated; the naive inverse filter's is not.
pad = size(psf,1);  Np = oddSize(N + 2*pad);
H = otf(psf, Np);
c = floor(Np/2) + 1;
[xx,yy] = meshgrid((1:Np)-c, (1:Np)-c);
rr = hypot(xx,yy);
fprintf('\nfraction of kernel energy within radius r:\n');
fprintf('   %-16s %8s %8s %8s %8s\n', 'filter', 'r<=8', 'r<=16', 'r<=32', 'r<=64');
Wall = [{[]}, num2cell(Ks)];
for i = 1:numel(Wall)
	if isempty(Wall{i})
		W = 1 ./ H;  lab = 'naive 1/H';
	else
		W = conj(H)./(abs(H).^2 + Wall{i});  lab = sprintf('Wiener K=%.0e', Wall{i});
	end
	w = real(fftshift(ifft2(W)));  tot = sum(w(:).^2);
	f = @(R) sum(w(rr<=R).^2)/tot;
	fprintf('   %-16s %8.4f %8.4f %8.4f %8.4f\n', lab, f(8), f(16), f(32), f(64));
end
fprintf(['   a kernel you can crop is one with its energy near the centre.\n' ...
	'   the naive filter spreads its energy everywhere, which is the real\n' ...
	'   reason the spatial route only works "sometimes".\n']);
end


function showSpatialExact(fignum, x, psf, sigma, K, P)
% Window 4: Wiener by FFT, and the same filter applied as an explicit spatial
% convolution with the FULL kernel. Everything here is circular: the blur, the
% FFT deconvolution and the spatial convolution all wrap the same way, so the
% two routes are the same operation and agree to machine precision.
% Truncating the kernel is what breaks that, and for a coded aperture no
% practical crop is close - which is why the spatial route is not used here.
if nargin < 6, P = 129; end
if mod(P,2)==0, P = P + 1; end
o = floor((size(x,1) - P)/2) + 1;
xp = x(o:o+P-1, o:o+P-1);                 % a patch, so the full kernel is affordable
 
H = otf(psf, P);
rng(0);
y = real(ifft2(fft2(xp) .* H)) + sigma*randn(P);
W = conj(H) ./ (abs(H).^2 + K);
exact = real(ifft2(W .* fft2(y)));
w = real(fftshift(ifft2(W)));
 
% the same filter, applied tap by tap in the spatial domain
c = floor(P/2) + 1;
acc = zeros(P);
for i = 1:P
	for j = 1:P
		if w(i,j) ~= 0
			acc = acc + w(i,j) * circshift(y, [i-c, j-c]);
		end
	end
end
 
d = acc - exact;
spOps  = P^4;
fftOps = 2*P*P*log2(P*P);
fprintf('\nfull-kernel spatial vs FFT, %dx%d patch, K = %.0e\n', P, P, K);
fprintf('   max |spatial - FFT| = %.2e   (image range %.2f to %.2f)\n', ...
	max(abs(d(:))), min(exact(:)), max(exact(:)));
fprintf('   same operation, machine precision apart\n');
fprintf('   cost: spatial %.2e ops, FFT %.2e ops  ->  %.0fx\n', spOps, fftOps, spOps/fftOps);
 
sfigure(fignum); clf;
ddsubplot(1,3,1); imagesc(y); axis image off; colormap(gca,'gray'); caxis([0 1]);
title('blurred + noise', 'FontSize', 14);
ddsubplot(1,3,2); imagesc(exact); axis image off; colormap(gca,'gray'); caxis([0 1]);
title('Wiener, via the FFT', 'FontSize', 14);
ddsubplot(1,3,3); imagesc(acc); axis image off; colormap(gca,'gray'); caxis([0 1]);
title(sprintf('same filter as a convolution\nidentical to %.0e', max(abs(d(:)))), ...
	'FontSize', 14);
drawnow;
end
 
function reportSpatialCost(Ns)
% Operation counts for a full-kernel spatial convolution against the FFT route.
fprintf('\nfull-kernel spatial convolution vs the FFT:\n');
fprintf('   %8s %12s %12s %10s\n', 'image', 'spatial', 'FFT', 'ratio');
for N = Ns
	sp = N^4;  ff = 2*N*N*log2(N*N);
	fprintf('   %5dx%-4d %12.2e %12.2e %9.0fx\n', N, N, sp, ff, sp/ff);
end
fprintf('   a compact kernel changes M^2 from N^2 to something small, which is\n');
fprintf('   the only way spatial wins - and for this filter no compact kernel\n');
fprintf('   is accurate enough.\n');
end
 
