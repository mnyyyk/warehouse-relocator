// frontend/src/components/SkuFlowCharts.tsx
import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  ArcElement,
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement,
  ArcElement
);

// ===== types =====
type SkuMetric = {
  sku_id: string;
  cases_per_day?: number;
  hits_per_day?: number;
  cube_per_day?: number;
  turnover_rate?: number;
  shipped_cases_all?: number;
  current_cases?: number;
  recv_cases_per_day?: number;
  recv_hits_per_day?: number;
  recv_cube_per_day?: number;
  updated_at?: string;
};

type ShipTxData = {
  sku_id: string;
  qty: number;
  trandate: string;
};

type RecvTxData = {
  sku_id: string;
  qty: number;
  trandate: string;
  lot?: string;
};

// ===== props =====
interface SkuFlowChartsProps {
  topMovers: SkuMetric[];
  shipData?: ShipTxData[];
  recvData?: RecvTxData[];
  selectedSku?: string;
  windowDays: number;
}

// ===== component =====
const SkuFlowCharts: React.FC<SkuFlowChartsProps> = ({
  topMovers,
  shipData = [],
  recvData = [],
  selectedSku,
  windowDays
}) => {
  // 1. 回転率ランキングチャート
  const turnoverChart = useMemo(() => {
    const top10 = topMovers
      .filter(m => (m.turnover_rate ?? 0) > 0)
      .slice(0, 10);

    return {
      labels: top10.map(m => m.sku_id),
      datasets: [{
        label: '在庫回転率',
        data: top10.map(m => m.turnover_rate ?? 0),
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      }]
    };
  }, [topMovers]);

  // 2. 流動性比較（出荷 vs 入荷）
  const flowComparisonChart = useMemo(() => {
    const top10 = topMovers.slice(0, 10);
    
    return {
      labels: top10.map(m => m.sku_id),
      datasets: [
        {
          label: '出荷頻度（cases/day）',
          data: top10.map(m => m.cases_per_day ?? 0),
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1,
        },
        {
          label: '入荷頻度（cases/day）',
          data: top10.map(m => m.recv_cases_per_day ?? 0),
          backgroundColor: 'rgba(75, 192, 192, 0.6)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1,
        }
      ]
    };
  }, [topMovers]);

  // 3. 特定SKUの時系列グラフ
  const timeSeriesChart = useMemo(() => {
    if (!selectedSku || (!shipData.length && !recvData.length)) {
      return null;
    }

    const skuShipData = shipData.filter(s => s.sku_id === selectedSku);
    const skuRecvData = recvData.filter(r => r.sku_id === selectedSku);

    // 日付別に集計
    const shipByDate = new Map<string, number>();
    const recvByDate = new Map<string, number>();

    skuShipData.forEach(s => {
      const date = s.trandate.split('T')[0];
      shipByDate.set(date, (shipByDate.get(date) ?? 0) + s.qty);
    });

    skuRecvData.forEach(r => {
      const date = r.trandate.split('T')[0];
      recvByDate.set(date, (recvByDate.get(date) ?? 0) + r.qty);
    });

    // 過去windowDays分の日付を生成
    const dates: string[] = [];
    const today = new Date();
    for (let i = windowDays - 1; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      dates.push(date.toISOString().split('T')[0]);
    }

    const shipValues = dates.map(date => shipByDate.get(date) ?? 0);
    const recvValues = dates.map(date => recvByDate.get(date) ?? 0);

    return {
      labels: dates,
      datasets: [
        {
          label: '出荷数',
          data: shipValues,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1,
        },
        {
          label: '入荷数',
          data: recvValues,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1,
        }
      ]
    };
  }, [selectedSku, shipData, recvData, windowDays]);

  // 4. 流動性分布（パイチャート）
  const flowDistributionChart = useMemo(() => {
    const ranges = [
      { label: '高流動（>10 cases/day）', min: 10, color: 'rgba(255, 99, 132, 0.8)' },
      { label: '中流動（1-10 cases/day）', min: 1, max: 10, color: 'rgba(54, 162, 235, 0.8)' },
      { label: '低流動（0.1-1 cases/day）', min: 0.1, max: 1, color: 'rgba(255, 206, 86, 0.8)' },
      { label: '静止（<0.1 cases/day）', max: 0.1, color: 'rgba(75, 192, 192, 0.8)' },
    ];

    const counts = ranges.map(range => {
      return topMovers.filter(m => {
        const casesPerDay = m.cases_per_day ?? 0;
        const minOk = range.min ? casesPerDay >= range.min : true;
        const maxOk = range.max ? casesPerDay < range.max : true;
        return minOk && maxOk;
      }).length;
    });

    return {
      labels: ranges.map(r => r.label),
      datasets: [{
        data: counts,
        backgroundColor: ranges.map(r => r.color),
        borderWidth: 1,
      }]
    };
  }, [topMovers]);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
      },
    },
  };

  const timeSeriesOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      title: {
        display: true,
        text: `${selectedSku} の流動性推移（過去${windowDays}日間）`,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: '日付'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'ケース数'
        }
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* 流動性サマリー */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
          <div className="text-red-800 font-semibold">高流動 SKU</div>
          <div className="text-2xl font-bold text-red-600">
            {topMovers.filter(m => (m.cases_per_day ?? 0) >= 10).length}
          </div>
          <div className="text-xs text-red-600">≥10 cases/day</div>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <div className="text-blue-800 font-semibold">中流動 SKU</div>
          <div className="text-2xl font-bold text-blue-600">
            {topMovers.filter(m => {
              const cpd = m.cases_per_day ?? 0;
              return cpd >= 1 && cpd < 10;
            }).length}
          </div>
          <div className="text-xs text-blue-600">1-10 cases/day</div>
        </div>
        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <div className="text-yellow-800 font-semibold">低流動 SKU</div>
          <div className="text-2xl font-bold text-yellow-600">
            {topMovers.filter(m => {
              const cpd = m.cases_per_day ?? 0;
              return cpd >= 0.1 && cpd < 1;
            }).length}
          </div>
          <div className="text-xs text-yellow-600">0.1-1 cases/day</div>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="text-gray-800 font-semibold">静止 SKU</div>
          <div className="text-2xl font-bold text-gray-600">
            {topMovers.filter(m => (m.cases_per_day ?? 0) < 0.1).length}
          </div>
          <div className="text-xs text-gray-600">&lt;0.1 cases/day</div>
        </div>
      </div>

      {/* グラフエリア */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 回転率ランキング */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            📊 在庫回転率ランキング（TOP10）
          </h3>
          <div style={{ height: '350px' }}>
            <Bar
              data={turnoverChart}
              options={{
                ...chartOptions,
                maintainAspectRatio: false,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: '在庫回転率（出荷数/在庫数）',
                  },
                },
              }}
            />
          </div>
        </div>

        {/* 流動性分布 */}
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            🍩 流動性分布
          </h3>
          <div style={{ height: '350px' }}>
            <Doughnut
              data={flowDistributionChart}
              options={{
                ...chartOptions,
                maintainAspectRatio: false,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'SKU流動性カテゴリ別分布',
                  },
                },
              }}
            />
          </div>
        </div>
      </div>

      {/* 流動性比較 */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          📈 出荷・入荷流動性比較（TOP10）
        </h3>
        <div style={{ height: '400px' }}>
          <Bar
            data={flowComparisonChart}
            options={{
              ...chartOptions,
              maintainAspectRatio: false,
              plugins: {
                ...chartOptions.plugins,
                title: {
                  display: true,
                  text: '日次平均ケース数',
                },
              },
            }}
          />
        </div>
      </div>

      {/* 特定SKUの時系列 */}
      {timeSeriesChart && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            📉 SKU別流動性推移
          </h3>
          <div style={{ height: '400px' }}>
            <Line 
              data={timeSeriesChart} 
              options={{
                ...timeSeriesOptions,
                maintainAspectRatio: false,
              }} 
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default SkuFlowCharts;