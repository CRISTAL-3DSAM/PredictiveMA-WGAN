function skel = denormalize(frame,path_mean,path_norm, path_center)
	CENTER = load(path_center);
	MEAN = load(path_mean);
	NORM = load(path_norm);
	CENTER = CENTER.centering;
	MEAN=MEAN.means;
	NORM=NORM.norms;
	skel = zeros(size(frame,1),size(frame,2));
	skel(:,1)= frame(:,1)+CENTER(1);
	skel(:,2)= frame(:,2)+CENTER(2);
	skel(:,3)= frame(:,3)+CENTER(3);
	skel = skel * NORM;
	skel(:,1)= skel(:,1)+MEAN(1);
	skel(:,2)= skel(:,2)+MEAN(2);
	skel(:,3)= skel(:,3)+MEAN(3);
end
	