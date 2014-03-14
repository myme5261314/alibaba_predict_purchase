function [ train_set, test_set ] = splitTrainTestSet( origin_path )
%SPLITTRAINTESTSET Summary of this function goes here
%   This is the function to split the raw data to the training set and test
%   data set. the training set is the record in the first three month, and
%   the test data set is the record in the 4th month.
if nargin == 0
    origin_path = 'alibaba_process.csv';
end
origin_data = csvread(origin_path);
%   Transfer the raw date to range [1-122].
origin_data(:, 4) = origin_data(:, 4) - min( origin_data(:,4)) + 1;
%   Sort the origin data rows by the record time with ASC order.
sorted_data = sortrows(origin_data, 4);
%   split the data record by whether it's record time is not less than 90
%   days. And extract the 1-90 days' data and the 91-122 days' data.
split_days = 60;
train_data = sorted_data(sorted_data(:,4)<=split_days, :);
test_data = sorted_data(sorted_data(:,4)>split_days, :);
train_set = sortrows( train_data, [1 2 4] );
test_set = sortrows( test_data, [1 2 4]);


    

end

