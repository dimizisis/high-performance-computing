import java.io.RandomAccessFile;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

class CharFreqParallelArrayOfLocks {

	static final int N = 128;
    static final int base = 0;
	public static void main(String[] args) throws IOException, InterruptedException {

		RandomAccessFile pFile;
        long fileSize;
        int numThreads;
		char[] buffer;
		String filename;
		int[] freq = new int[N];
        long start, end;
        Lock[] locks = new ReentrantLock[N];

		if(args.length != 2) {
			System.out.println("Usage : CharFreqSeq <file_name> <numThreads>");
			System.exit(1);
			return;
        }
        
        filename = args[0];
        numThreads = Integer.parseInt(args[1]);
        
		try {
			pFile = new RandomAccessFile(filename.toString(), "r");
		} catch(IOException ex) {
            System.out.println("File error");
			System.exit(2);
			return;
		}

		// obtain file size:
		pFile.seek(pFile.length());
		fileSize = pFile.getFilePointer();
		pFile.seek(0);
		System.out.println("file size is " + fileSize);

		byte[] byteArray = new byte[(int)fileSize];
		pFile.readFully(byteArray);

        // allocate memory to contain the file:
        buffer = new String(byteArray, StandardCharsets.UTF_8).toCharArray();

        CountThread[] threads = new CountThread[numThreads];

        int blocks = Math.toIntExact(fileSize) / numThreads;

        /* Initialize locks */
        for(int i=0; i<N; ++i)
            locks[i] = new ReentrantLock();

        /* Create threads */
        for (int i=0, j=0; i<numThreads; ++i, j+=blocks)
            threads[i] = new CountThread(freq, buffer, locks, i, j, blocks);

        start = System.currentTimeMillis();
        
        /* Start all threads */
        for (int i=0; i<numThreads; ++i)
            threads[i].start();

        /* Wait for all threads to finish */
        for (int i=0; i<numThreads; ++i)
            threads[i].join();

		end = System.currentTimeMillis();

		displayCount(freq, N);

		System.out.printf("Time spent for counting: %g", (double)(end - start) / 1000);

		pFile.close();
    }

    public static void displayCount(int[] freq, int n) {
		for(int i=0; i<n; ++i)
			System.out.println(i + " = " + freq[i]);
	}

    private static class CountThread extends Thread {

        private int[] freq;
        private char[] buffer;
        private Lock[] locks;
        private int tid;
        private int idx;
        private int blocks;

        public CountThread(int[] freq, char[] buffer, Lock[] locks, int tid, int idx, int blocks) {
            this.freq = freq;
            this.buffer = buffer;
            this.locks = locks;
            this.tid = tid;
            this.idx = idx;
            this.blocks = blocks;
        }

        @Override
        public void run(){
            int stop = tid*blocks + blocks;
            for (int i=idx; i<stop; ++i){
                locks[buffer[i] - base].lock();
                ++freq[buffer[i] - base];
                locks[buffer[i] - base].unlock();
            }
        }
    }

}