
import java.util.concurrent.*;

class PiMonteCarloParallelForkJoin {
    
    public static void main(String[] args) {

        if (args.length != 2) {
            System.out.println("Usage : PiMonteCarloParallelForkJoin <num points> <threshold>");
          System.exit(1);
        }
        int niter = Integer.parseInt(args[0]);
        int threshold = Integer.parseInt(args[1]);

        /* Create forkjoin pool. 'Common pool' uses all available processors */
        ForkJoinPool pool = ForkJoinPool.commonPool();

        /* Create initial task */
        CalculationTask task = new CalculationTask(0, niter, threshold);

        /* Return count value */
        int count = pool.invoke(task);
        
        /* Shutdown forkjoin pool */
        pool.shutdown();

        /* Final calculation (pi) */
        double pi = (double) count/niter*4;

        /* Print results */
        System.out.println(pi);

    }

    private static class CalculationTask extends RecursiveTask<Integer> {

        private int start;
        private int stop;
        private int threshold;
        private int count;

        public CalculationTask(int start, int stop, int threshold) {
            this.start = start;
            this.stop = stop;
            this.threshold = threshold;
        }

        @Override
        protected Integer compute() {

            int workLoadSize = stop - start;
            if (workLoadSize <= threshold) { /* Base case */
                computeDirectly();
                return count;
            }
            
            /* Find middle index (in order to divide problem) */
            int mid = (start+stop)/2;
            
            /* Divide the problem */
            CalculationTask left = new CalculationTask(start, mid, threshold);    
            CalculationTask right = new CalculationTask(mid, stop, threshold);
            /* ****************** */

            /* InvokeAll submits a sequence of ForkJoinTasks to the ForkJoinPool */
            invokeAll(left, right);

            /***  instead of invokeAll usage:

            left.fork();
            double rightResult = right.compute();
            double leftResult = left.join();
            sum = rightResult + leftResult;

            *** ************************ ***/
            
            /* Trigger execution of left & right tasks*/
            count = left.join() + right.join();

            return count;

        }

        protected void computeDirectly(){
            double x, y, z;
            for (int i=start; i < stop; ++i) {
                x = Math.random();
                y = Math.random();
                z = x*x+y*y;
                if(z<=1)
                    ++count;
            }
        }
    }

}
