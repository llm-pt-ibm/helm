import { useEffect, useState } from "react";
import getStatsByName from "@/services/getStatsByName";
import type Stat from "@/types/Stat";
import Loading from "@/components/Loading";

const TABLE_COLUMN_NAMES = ["method", "exact_match", "rouge_l"];

interface Props {
  runName: string;
  suite: string;
}

export default function ContaminationMetrics({ runName, suite }: Props) {
  const [stats, setStats] = useState<Stat[] | undefined>();

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const signal = controller.signal;
      const statsResp = await getStatsByName(runName, signal, suite);
      setStats(statsResp);
    }

    void fetchData();

    return () => controller.abort();
  }, [runName, suite]);

  if (stats === undefined || stats.length === 0) {
    return <Loading />;
  }

  // Filter only contamination metrics
  const contaminationStats = stats.filter(
    (stat) => stat.name.name === "contamination",
  );

  if (contaminationStats.length === 0) {
    return (
      <div className="alert alert-info">
        <div>
          <span>No contamination metrics found for this run.</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              {TABLE_COLUMN_NAMES.map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {contaminationStats.map((stat, index) => (
              <tr key={index}>
                {TABLE_COLUMN_NAMES.map((key) => {
                  if (key === "method") {
                    return <td key={key}>{stat.method}</td>;
                  } else if (key === "exact_match" || key === "rouge_l") {
                    const value = stat[key];
                    return (
                      <td key={key}>
                        {typeof value === "number" ? value : "-"}
                      </td>
                    );
                  } else {
                    return <td key={key}>-</td>;
                  }
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
