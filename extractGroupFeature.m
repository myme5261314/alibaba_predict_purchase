function [ X, Y ] = extractGroupFeature( data_set )
%EXTRACTGROUPFEATURE Summary of this function goes here
%   This is the function to extract the record with the same (user_id,
%   brand_id) tuple and summarize the records to generate the features.
user_brand_keys = unique(data_set(:,[1 2]), 'rows');
extractFeatures = [];
for i = 1:length(user_brand_keys)
    user_brand = user_brand_keys(i,:);
    ind = bsxfun(@and,data_set(:,1) == user_brand(1), ...
        data_set(:,2) == user_brand(2));
    one_group = data_set(ind,:);
    click_num = sum(one_group(:,3)==0);
    collect_num = sum(one_group(:,3)==1);
    addtocart_num = sum(one_group(:,3)==2);
    purchase_num = sum(one_group(:,3)==3);
    oneFeature = [ user_brand click_num collect_num addtocart_num...
        purchase_num];
    extractFeatures = [extractFeatures; oneFeature];
end
X = extractFeatures(:,1:5);
Y = extractFeatures(:,6);

end

