import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ModelPerformanceChart = () => {
  // Extracted data from the benchmark logs
  const data = [
    { points: 10000, "Distil-BERT": 87.03, "BioClinicalBERT": 116.67, "Clinical ModernBERT": 72.19 },
    { points: 20000, "Distil-BERT": 174.99, "BioClinicalBERT": 230.63, "Clinical ModernBERT": 151.42 },
    { points: 30000, "Distil-BERT": 261.87, "BioClinicalBERT": 353.38, "Clinical ModernBERT": 227.37 },
    { points: 40000, "Distil-BERT": 342.14, "BioClinicalBERT": 469.00, "Clinical ModernBERT": 302.69 },
    { points: 50000, "Distil-BERT": 433.92, "BioClinicalBERT": 588.50, "Clinical ModernBERT": 370.72 },
    { points: 60000, "Distil-BERT": 523.46, "BioClinicalBERT": 707.57, "Clinical ModernBERT": 462.22 },
    { points: 70000, "Distil-BERT": 614.15, "BioClinicalBERT": 828.94, "Clinical ModernBERT": 532.25 },
    { points: 80000, "Distil-BERT": 679.98, "BioClinicalBERT": 947.71, "Clinical ModernBERT": 603.95 },
    { points: 90000, "Distil-BERT": 761.55, "BioClinicalBERT": 1044.25, "Clinical ModernBERT": 654.87 },
    { points: 100000, "Distil-BERT": 853.28, "BioClinicalBERT": 1178.38, "Clinical ModernBERT": 729.29 }
  ];

  return (
    <div className="w-full h-96 p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold text-center mb-4">BERT Model Processing Time Comparison</h2>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart 
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 30 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="points" 
            label={{ value: 'Number of Data Points', position: 'bottom', offset: 0 }} 
          />
          <YAxis 
            label={{ value: 'Processing Time (seconds)', angle: -90, position: 'insideLeft' }} 
          />
          <Tooltip formatter={(value) => [`${value.toFixed(2)} seconds`, null]} />
          <Legend verticalAlign="top" />
          <Line 
            type="monotone" 
            dataKey="Distil-BERT" 
            stroke="#8884d8" 
            strokeWidth={2} 
            activeDot={{ r: 8 }} 
          />
          <Line 
            type="monotone" 
            dataKey="BioClinicalBERT" 
            stroke="#ff7300" 
            strokeWidth={2} 
            activeDot={{ r: 8 }} 
          />
          <Line 
            type="monotone" 
            dataKey="Clinical ModernBERT" 
            stroke="#82ca9d" 
            strokeWidth={2} 
            activeDot={{ r: 8 }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ModelPerformanceChart;
