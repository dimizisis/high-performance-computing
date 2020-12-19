
import java.util.Random;
class CountSortParallelSharedArray {
    public static void main(String[] args) throws InterruptedException {

        if (args.length != 2) {
            System.out.println("Usage : CountSortParallelSharedArray <num_elements> <num_threads>");
            System.exit(1);
        }
        int n = Integer.parseInt(args[0]);
        int numThreads = Integer.parseInt(args[1]);

        int[] array = new int[n];
        int[] sortedArray = new int[n];

        randomArray(array, n);

        /* Print initial array */
        System.out.print("Initial Array: [");
        for (int i=0; i<n; ++i)
            System.out.print(" " + array[i]);
        System.out.println(" ]");

        int blocks = n / numThreads;

        SortingThread[] threads = new SortingThread[numThreads];
        
        /* Create threads */
        for(int i=0, j=0; i<numThreads; ++i, j+=blocks)
            threads[i] = new SortingThread(array, sortedArray, n, j, i, blocks);

        /* Start threads */
        for (int i=0; i<numThreads; ++i)
            threads[i].start();

        /* Wait for all threads to finish */
        for (int i=0; i<numThreads; ++i)
            threads[i].join();
        
        /* Print results */
        System.out.print("Sorted Array: [");
        for (int i=0; i<n; ++i)
            System.out.print(" " + sortedArray[i]);
        System.out.println(" ]");
    }

    private static void randomArray(int[] array, int n){
        int i;    
        for (i=0; i<n; ++i)
            array[i] = new Random().nextInt(100);
    }

    private static class SortingThread extends Thread {

        private int[] sortedA;
        private int[] a;
        private int n;
        private int blocks;
        private int start;
        private int rank;

        public SortingThread(int[] a, int[] sortedA, int n, int start, int rank, int blocks){
            this.a = a;
            this.sortedA = sortedA;
            this.n = n;
            this.start = start;
            this.rank = rank;
            this.blocks = blocks;
        }

        @Override
        public void run(){
            int count;
            int stop = rank*blocks + blocks;
            for (int i = start; i < stop; ++i) {
                count = 0;
                for (int j = 0; j < n; ++j)
                    if (a[j] < a[i])
                        ++count;
                    else if (a[j] == a[i] && j < i)
                        ++count;
                sortedA[count] = a[i];
            }
        }
        
    }
}