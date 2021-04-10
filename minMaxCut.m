%% Setup 2D model
clear all; close all; clc;

%% DEFINE GRID AND GEOSTATISTICAL PARAMETERS
% modelName = 'perm_20x20.GRDECL';
modelName = 'SPE10';
nx = 20; ny = 20; nz = 1;
dx = 50; dy = 50; dz = 100;
r = 100; % Radius of influence
mode = 1;  % 1 for spherical spatial correlation; and 2 for ellipsoildal
xAxis = 400;  yAxis = 100;  zAxis = 50;
theta1 = 45; theta2 = 0; theta3 = 0;  %ORIENTATION: (1) for azimuth, (2) for dip and (3) for rake

% Graph Size
NV = nx*ny*nz;

%% READ PROPERTY
if ~strcmp(modelName,'SPE10')
    fid = fopen(modelName,'r');
    k = zeros(NV,1);
    for line = 1 : (NV+100)
        label = fgetl(fid);
        if(strcmp(label,'PERMX') == 1)
            for i = 1 : NV
                k(i) = fscanf(fid,'%f',1);
            end
        end
    end
    fclose(fid);
else
    load grdecl_model
    permx_mat = reshape(PERM(:,3),60,220,85);
    k1 = permx_mat(:,:,50); k = k1(:);
    nx = 60; ny = 220; nz = 1;
    dx = 20; dy = 10; dz = 2;
    NV = nx*ny*nz;
    r = 300; % Radius of influence
end
logk = log(k);
figure(1);  imagesc(reshape(logk,nx,ny));
xlabel('X'); ylabel('Y'); 
% set(gca,'YDir','normal')
colorbar; saveas(gcf,'logk','emf')


%% PROCESS: Assuming uniform gridding
xCen1 = ((1:nx)'-0.5)*dx;
yCen1 = ((1:ny)'-0.5)*dy;
zCen1 = ((1:nz)'-0.5)*dz;

xCen = repmat(xCen1,[1,ny,nz]); xCenVec = xCen(:);
yCen = repmat(yCen1',[nx,1,nz]); yCenVec = yCen(:);
zCen = permute(repmat(zCen1,[1,ny,nx]),[3,2,1]); zCenVec = zCen(:);

% COORDIANTE TRANSFORMATION
if mode ==2
    T3 = [1 0 0; 0 cos(theta3/180*pi) sin(theta3/180*pi); 0 -sin(theta3/180*pi) cos(theta3/180*pi)];    %Rake transformation
    T2 = [cos(theta2/180*pi) 0 sin(theta2/180*pi); 0 1 0; -sin(theta2/180*pi) 0 cos(theta2/180*pi)];    %Dip transformation
    T1 = [cos(theta1/180*pi) sin(theta1/180*pi) 0; -sin(theta1/180*pi) cos(theta1/180*pi) 0; 0 0 1];    %Azimuth transformation
    T = T3*T2*T1;
end


%% ADJACENCY PARAMETERS
epsilon = 0.001; 
dP = 0.02*sqrt(range(k)); 
sigP = -dP^2/log(epsilon);
sigD = -r^2/log(epsilon);



%% GET CONNECTIVITY
GridCon = zeros(NV);
Dist = zeros(NV);
Prop = zeros(NV);
grdVec = [xCenVec yCenVec zCenVec];
for i = 1:NV
    posVec = repmat([xCenVec(i) yCenVec(i) zCenVec(i)],NV,1);
    if mode==2;
        transDist = T*(posVec - grdVec)';
        dVec = sum((transDist').^2./repmat([xAxis yAxis zAxis].^2,NV,1),2);
    else
        dVec = sqrt(sum((posVec - grdVec).^2,2))/r;
    end
    sigD = 1;
    [dummy ind] = sort(dVec);
    nInd = find(dummy<=1,1,'last');
    cellinds = ind(2:nInd)';
    GridCon(i,1:nInd-1) = cellinds;
    Dist(i,cellinds) = dummy(2:nInd)';
    Prop(i,cellinds) = abs(logk(i)-logk(cellinds))';
end
    

%% GET ADJACENCY AND LAPLACIAN
A = zeros(NV); D = zeros(NV); L = zeros(NV);
A(Dist~=0) = exp(-Prop(Dist~=0)/sigP) .* exp(-Dist(Dist~=0)/sigD);
diagVec = sum(A,2);
D(1:length(D)+1:numel(D)) = diagVec;
L = D-A;


%% EIGENDECOMPOSITION AND SPECTRAL CLUSTERING
tic; [u s v] = eigs(sparse(L),20,'sm'); toc
k = 5;
U = fliplr(u); [ID] = kmeans(U(:,1:k),k);
figure(2);imagesc(reshape(ID,nx,ny))


%% GET OBJECTIVE FUNCTION
func = @(x,L) trace(x'*L*x);
for i = 1:k; 
    h = zeros(NV,1);
    h(ID==i) = 1;
    H1(:,i) = h; 
    H(:,i) = h/sqrt(sum(h)); 
end   
OF = feval(func,H,L)


%% LOCAL SEARCH
[ID] = randi(k,nx*ny,1);
figure(3);imagesc(reshape(ID,nx,ny))
H1 = zeros(NV,k); H = zeros(NV,k); 
for i = 1:k; 
    h = zeros(NV,1);
    h(ID==i) = 1;
    H1(:,i) = h; 
    H(:,i) = h/sqrt(sum(h)); 
end    
% 
%Initial Objective function
OF = feval(func,H,L);
OF_vec = OF;

%Begin iteration
nitr = 50000;
sampleInd = randi(k,nitr,1);
sampleFac = rand(nitr,1);
 for i = 1:nitr
    H1test = H1;
    k_col = sampleInd(i);
    indvec = find(H(:,k_col)==0);
    j = round(sampleFac(i)*numel(indvec)); if j == 0; j = 1; end   
    ind2one = indvec(j); %Randomly choose one of the vertices outside the k_col^th cluseter
    H1test(ind2one,k_col) = 1; % Add vertex from ind2zero^th cluster to k_col^th cluster
    ind2zero = find(H1(ind2one,:)==1);
    H1test(ind2one,ind2zero) = 0;   % Remove vertex from ind2zero^th cluster
    Htest = H1test./repmat(sqrt(sum(H1test)),NV,1);
    OFtest = feval(func,Htest,L);
    if OFtest < OF

        
        H1 = H1test; H = Htest;
        OF = OFtest; 
    end
    OF_vec = [OF_vec; OF];  
 end

ID_OP = k*ones(NV,1);
for i = 1:k-1
    ID_OP(find(H1(:,i)==1)) = i;
end
figure(4); imagesc(reshape(ID_OP,nx,ny));
figure(5); plot(OF_vec);



% %% L-K LOCAL SEARCH
% [H1_best, G_best, G, g_vec] = LK_search(A, k, NV, ID);
% 
% ID_OP = k*ones(NV,1);
% for i = 1:k-1
%     ID_OP(find(H1_best(:,i)==1)) = i;
% end
% figure(4); imagesc(reshape(ID_OP,nx,ny));
% H_best = H1_best./repmat(sqrt(sum(H1_best)),NV,1);
% improvement = feval(func,H_best,L) - feval(func,H,L);



% %% GA
% 
% % Specify Parameters
% nbits = floor(log2(k))+1;
% npop = 40;
% ngen = 50;
% mut = 'true';
% CO = 'true';
% localSearch = 'true';
% CO_pts = 2;
% mProb = 0.70;  %Mutation Probability
% truncFac = 0.3;
% if truncFac > 0.5; error('maximum subset factor is 0.5'); end
% 
% % Initialize 
% sol = randi(k,NV,npop);
% if localSearch
%     for ilk = 1:npop
%         [H1_best, G_best, G, g_vec] = LK_search(A, k, NV, sol(:,ilk));
%         ID_OP = k*ones(NV,1);
%         for i = 1:k-1
%             ID_OP(find(H1_best(:,i)==1)) = i;
%             sol(:,ilk) = ID_OP;
%         end
%     end
% end
%     
% sol_bin = genBinIntConv(sol, nbits, k, 1);
% all_fitness = [];
% 
% %Calculate initial fitness
% fitness = evalFit(sol, L, k);
% all_fitness = [all_fitness fitness];
% [dum,ind] = sort(fitness); fitness = dum; clear dum;
% dum = sol_bin(:,ind); sol_bin = dum; clear dum;  % Sort current solutions in current generation
% 
% % Run
% for i = 2:ngen
%     % Tournament select
%    if CO 
%         select_size = 2*floor(truncFac*npop);
%    else
%        select_size = floor(truncFac*npop);
%    end
%    
%    if CO
%        % Determine mating schedule if CrossOver is required
%        unrealistic = 1;
%        while unrealistic
%            n = select_size;
%            schd = randi(n,n/2,2);
%            unrealistic = any(schd(:,1)-schd(:,2)==0); 
%        end
%        %Perform mating
%        new_sol_bin = [];
%        point = sort(randi(size(sol_bin,1),CO_pts,size(schd,1)),1);
%        point = [ones(1,size(point,2)); point; size(sol_bin,1)*ones(1,size(point,2))];
%        for j = 1:size(schd,1) 
%            parent1 = sol_bin(:,schd(j,1));
%            parent2 = sol_bin(:,schd(j,2));
%            child1 = parent1; child2 = parent2;
%            for jj = 2:2:CO_pts+1
%                 child1(point(jj,j):point(jj+1,j)) = child2(point(jj,j):point(jj+1,j));
%                 child2(point(jj,j):point(jj+1,j)) = child1(point(jj,j):point(jj+1,j));
%            end
%            new_sol_bin = [new_sol_bin child1 child2]; 
%        end
%    else
%        new_sol_bin = sol_bin(:,1:select_size);
%    end
%    
%    %Mutation
%    if mut
%        prbMat = rand(size(new_sol_bin));
%        flipMat = zeros(size(new_sol_bin));
%        flipMat(prbMat>=mProb) = 1;
%        new_sol_bin = abs(new_sol_bin - flipMat);
%    end
% 
%    % Evaluate fitness of new generation
%    new_sol = genBinIntConv(new_sol_bin, nbits, k, 2);
%    %Local Search if desired
%    if localSearch
%        for ilk = 1:size(new_sol,2)
%            [H1_best, G_best, G, g_vec] = LK_search(A, k, NV, new_sol(:,ilk));
%            ID_OP = k*ones(NV,1);
%            for ii = 1:k-1
%                ID_OP(find(H1_best(:,ii)==1)) = ii;
%                 new_sol(:,ilk) = ID_OP;
%            end
%        end
%    end
%    new_fit = evalFit(new_sol, L, k);
%    
%    genpool = [new_sol sol];
%    fitpool = [new_fit; fitness];
%    [dum,ind] = sort(fitpool);
%    fitness = fitpool(ind(1:npop));
%    sol = genpool(:,ind(1:npop)); 
%    sol_bin = genBinIntConv(sol, nbits, k, 1);
%    all_fitness = [all_fitness fitness];
%    
%    %Visualize new results
%    figure(10);
%    xaxis = repmat((1:i),npop,1);
%    plot(xaxis(:), all_fitness(:), 'o','markerfacecolor','r');
%    grid on; hold on
%    xlabel('Generations'); ylabel('Fitness');
%    hold off
% end    


 


    
    
    
    
    
    
    




