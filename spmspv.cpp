#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread> 
#include <fstream>
#include <math.h>


#include <xmmintrin.h>
#include <pmmintrin.h>

using namespace std::chrono;
using namespace std;

// ****************     PROBLEM VARIABLES/CONSTANTS       ****************

// define global variables that are shared between functions
const int max_dim = 5000;
#define THREAD_NUM 8


int n, m;
float matrix[max_dim+10][max_dim+10], v[max_dim+10], result[max_dim+10];


const string evaluation_file_name = "evaluation_data.csv";
// *************************************************************



// **************     ALGORITHMS VARIABLES       ***************

// coo format variables (allocate max possible size for row and columns)
int row_coo[(max_dim+10)*(max_dim+10)], column_coo[(max_dim+10)*(max_dim+10)];

// csr format variables
int row_offset[max_dim+10], columns[(max_dim+10)*(max_dim+10)];

// csc format variables
int column_offset[(max_dim+10)], rows[(max_dim+10)*(max_dim+10)];

// shared between coo, csr, csc  
float val[(max_dim+10)*(max_dim+10)]; 

// ell format variables
// initialize column_mat to -1, as its rows represent index of column, -1 is an invalid and specify that this row is empty!
int column_mat[(max_dim+10)][(max_dim+10)] {-1};
float ell_val[(max_dim+10)][(max_dim+10)];


// csr simd
float columns_simd[(max_dim+10)*(max_dim+10)], val_simd[(max_dim+10)*(max_dim+10)];

// ell simd
float column_mat_simd[(max_dim+10)][(max_dim+10)], ell_val_simd[(max_dim+10)][(max_dim+10)];
// *************************************************************




void generate_random_input( int  matrix_sparsity_factor = 60, int v_sparsity_factor = 10, int max_val= 1000, int max_dcp = 2 ){

    float dc_d = pow(10, max_dcp);

    string str {"Lehmer"};
    seed_seq seed1 (str.begin(),str.end());
    minstd_rand0 generator(seed1);
    

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            if (rand()% 100 > matrix_sparsity_factor)
                matrix[i][j] =((float)(generator() % (max_val *(int)dc_d))) / dc_d;   
            else
                matrix[i][j] = 0.0;   
        }
        
    }

    for (size_t i = 0; i < m; ++i)
       { if (rand()% 100 > v_sparsity_factor)
            v[i] = ((float)(generator() % (max_val *(int)dc_d))) / dc_d;
        else
            v[i] = 0.0;
       } 
}


void print_initial_matrix()
{
    cout<< "--------- Matrix ---------"<<endl;
    for(size_t i=0; i< n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
            if (j == 0)
                cout<< matrix[i][j];
            else
                cout<< '\t'<< matrix[i][j];
    
        cout<<endl;    
    }

    cout<< "--------- Vector ---------"<<endl;
      for (size_t j = 0; j < m; ++j)
        cout<< v[j]<<endl;

}

void print_result()
{
    cout<<"--------- result ---------"<<endl;
    for (size_t i = 0; i < n; ++i)
        cout<<result[i]<<'\t';
}


/* 
    ********************************************************************
                                SINGLE THREAD
    ******************************************************************** 
*/


/* 
    ********************************************************************
    COO implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/

int coo_constructor()
{
    fill(row_coo, row_coo + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(column_coo, column_coo + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(val, val + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(result, result + ((max_dim+10)), 0.0);


    int next_empty_index = 0; // store index of the first empty cell of the three defined array  
    
    // now fill x_coo,y_coo and val arrays' values respected to given matrix
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
        // cout<<"["<<i<<"]["<<j<<"]: "<<matrix[i][j]<<"\t"<<v[j]<<endl;

            if (matrix[i][j]!=0.0 && v[j] !=0.0)
            {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 
                // cout<<"in"<<endl;
                row_coo[next_empty_index] = i;
                column_coo[next_empty_index] = j;
                val[next_empty_index] = matrix[i][j];
                next_empty_index ++;
            }
            
        }
    }
    /*
    cout<< "------------ coo format result ------------\n";
    for (size_t i = 0; i < next_empty_index; i++)
    {
        cout<<"("<<row_coo[i]<<", "<<column_coo[i]<<"): "<< val[i]<<endl;
    }
     
    */

    return next_empty_index;

}

float coo_multi()
{
    // first convert matrix to a COO(coordinate) Format
    // time_req = clock();
    
    int next_empty_index = coo_constructor();
    
    // time_req = clock();
    
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < next_empty_index; ++i)
        result[row_coo[i]] += val[i] * v[column_coo[i]]; 

    auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);
    return (float)duration.count();
	// cout << "coo "<< duration2.count() << " micros" << endl;

    // time_req = clock() - time_req;
    // cout<<(float)time_req<<endl;
    // return (float)time_req;


}


/*
    ********************************************************************
    CSR implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/



int  csr_constructor()
{   
    fill(row_offset, row_offset + ((max_dim+10)), 0.0);
    fill(columns, columns + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(val, val + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(result, result + ((max_dim+10)), 0.0);
    
    // first index of the empty cell in columns and val arrays
    int first_empty_index = 0;
    
    for (size_t i = 0; i < n; ++i)
    {   
        int starts_from = first_empty_index;
        for (size_t j = 0; j < m; ++j)
        {
            if(matrix[i][j] != 0.0 && v[j] != 0.0)
            {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 
        
                columns[first_empty_index] = j;
                val[first_empty_index] = matrix[i][j];
                // move the index to the first empty cell in the columns and val arrays to next cell 
                // (as they have been filled by one value)
                first_empty_index ++;

            }
        }

        if(starts_from != first_empty_index) // this row is not empty
        {
            // add the starting point of first nonzero value of this row index in the columns and val array
            row_offset[i] = starts_from;
        }
        else
            row_offset[i] = -1;  // an invalid index to show that this row is empty
        
    }

    /*
    cout<< "------------ row offset ------------"<<endl;
    for (size_t i = 0; i < n; i++)
        cout<< row_offset[i]<< ' ';
    cout<< endl;

    cout<< "------------ columns ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< columns[i]<< ' ';
    cout<< endl;

    cout<< "------------ values ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< val[i]<< ' ';
    cout<< endl;
    */

    return first_empty_index;
}

float csr_multi()
{
    int first_empty_index = csr_constructor();


    auto start = high_resolution_clock::now();

    int upper_bound, skip_rows;

    for (size_t i = 0; i < n; ++i)
    {
        
        skip_rows = 0; 
        // first skip empty rows from here, to find exactly where to stop considering columns values
        while(row_offset[i + skip_rows + 1] == -1)
            skip_rows ++;
        
        if(i + 1 + skip_rows == n)
            upper_bound = first_empty_index;
        else if (row_offset[i] == -1) // current row row is empty, skip it
        {
            continue; // go and run the loop for the next i 
        }
        
        else
            upper_bound = row_offset[i+ skip_rows + 1];
        
        
        for (size_t j = row_offset[i]; j < upper_bound; ++j)
        {
            result[i] += val[j]*v[columns[j]];
        }
        // jump to the first row that is not empty
        i+= skip_rows;
    }   

    auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();


}


/*
    ********************************************************************
    CSC implementation of Sparse matrix for faster SpMSpV multiplication 
    ********************************************************************
*/

int  csc_constructor()
{
    fill(column_offset, column_offset + ((max_dim+10)), 0.0);
    fill(rows, rows + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(val, val + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(result, result + ((max_dim+10)), 0.0);
    
    // first index of the empty cell in rows and val arrays
    int first_empty_index = 0;
    
    for (size_t j = 0; j < m; ++j)
    {   
        int starts_from = first_empty_index;
        for (size_t i = 0; i < n; ++i)
        {
            if(matrix[i][j] != 0.0 && v[j] != 0.0)
            {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 

                rows[first_empty_index] = i;
                val[first_empty_index] = matrix[i][j];
                // move the index to the first empty cell in the rows and val arrays to next cell 
                // (as they have been filled by one value)
                first_empty_index ++;

            }
        }

        if(starts_from != first_empty_index) // this row is not empty
        {
            // add the starting point of first nonzero value of this row index in the rows and val array
            column_offset[j] = starts_from;
        }
        else
            column_offset[j] = -1; //mark this column with an invalid index ti identify that it is empty
        
    }

    /*
    cout<< "------------ columns offset ------------"<<endl;
    for (size_t i = 0; i < m; i++)
        cout<< column_offset[i]<< ' ';
    cout<< endl;

    cout<< "------------ rows ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< rows[i]<< ' ';
    cout<< endl;

    cout<< "------------ values ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< val[i]<< ' ';
    cout<< endl;
    */

    return first_empty_index;
}

float csc_multi()
{
    int first_empty_index = csc_constructor();

    
    auto start = high_resolution_clock::now();
    int upper_bound, skip_columns;


    for (size_t j = 0; j < n; ++j)
    {
        skip_columns = 0; 
        
        // first skip empty columns from here, to find exactly where to stop considering rows values
        while( column_offset[j + skip_columns + 1] == -1)
            skip_columns ++;
        
        if(j + 1 + skip_columns == m)
            upper_bound = first_empty_index;
        else if (column_offset[j] == -1)
        {
            continue; // this row is empty, run the loop for next column
        }
        
        else
            upper_bound = column_offset[j+ skip_columns + 1];
        
        
        for (size_t i = column_offset[j]; i < upper_bound; ++i)
        {
            result[rows[i]] += val[i]*v[j];
        }

        // jump to the first column that is not empty
        j+= skip_columns;
    }   

    auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();


    
}

/*
    ********************************************************************
    ELL implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/

int ell_constructor()
{
    fill(result, result + ((max_dim+10)), 0.0);
    fill(&column_mat[0][0], &column_mat[(max_dim+10)-1][0]+(m), -1);
    fill(&ell_val[0][0], &ell_val[((max_dim+10))-1][0]+((max_dim+10)), 0.0);
    
    int row_max = 0, tmp_row_max;
    for (size_t i = 0; i < n; ++i)
    {
        tmp_row_max = 0;
        for (size_t j = 0; j < m; ++j)
        {   
            if(matrix[i][j] != 0.0 && v[j] != 0.0)
            {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 

                column_mat[i][tmp_row_max] = j;
                ell_val[i][tmp_row_max] = matrix[i][j];
                tmp_row_max +=1;
            }

        }
        if(row_max < tmp_row_max)
            row_max = tmp_row_max;
        
    }

    /*
    cout<< "------------ column ------------\n";
    for (size_t i = 0; i < n; i++)
    {
        for(size_t j =0; j< row_max; j++)
        {
            cout<<col_mat[i][j] << ' ';
        }
        cout<<'\n';
    }
    cout<< '\n';

    cout<< "------------ values ------------\n";
    for (size_t i = 0; i < n; i++)
    {
        for(size_t j =0; j< row_max; j++)
        {
            cout<<val[i][j] << ' ';
        }
        cout<<'\n';
    }
    cout<<'\n';
    */
    return row_max;
    
}

float ell_multi()
{
    int max_row = ell_constructor(); 


    auto start = high_resolution_clock::now();
    
    int row_index;    
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < max_row; ++j)
        {
            row_index = column_mat[i][j];
            if (row_index != -1)
            {
                result[i] += ell_val[i][j] * v[row_index];
            }
            else
            {   // the rest of this row is emtpy, go tho the next row 
                // (because we filled column_mat form left, till there is no row tho add)  
                break;
            }
        }
        
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();

}




/* 
    ********************************************************************
                                MULTI THREAD
    ******************************************************************** 
*/


/* 
    ********************************************************************
    PArallel COO implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/

int coo_constructor_parallel(int threads_next_empty_index[][2])
{
    fill(row_coo, row_coo + (m*n), 0.0);
    fill(column_coo, column_coo + (m*n), 0.0);
    fill(val, val + (m*n), 0.0);
    fill(result, result + (n), 0.0);

    int first_row, last_row, row_batch_size;
    // if it should be run in single thread, return the actual next_empty index, if it should not, return -1 
    bool single_thread = false;
    int first_last_index[THREAD_NUM][2];

    if(n < THREAD_NUM){
        single_thread = true;
    }else
    {
        row_batch_size = n/THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; ++i) 
        {    
            switch (i)
            {
            case 0:
                first_row =0;
                last_row = row_batch_size;
                break;
            case THREAD_NUM:
                first_row = row_batch_size * (i);
                last_row = n;
                break;
            default:
                first_row = row_batch_size * (i);
                last_row = first_row + row_batch_size; 
                break;
            }
            first_last_index[i][0]= first_row;
            first_last_index[i][1]= last_row;

        }
    }



    int next_empty_index = 0; // store index of the first empty cell of the three defined array  
    
    // now fill x_coo,y_coo and val arrays' values respected to given matrix
    if(single_thread)
    {
           for (size_t i = 0; i <n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
            // cout<<"["<<i<<"]["<<j<<"]: "<<matrix[i][j]<<"\t"<<v[j]<<endl;

                if (matrix[i][j]!=0.0 && v[j] !=0.0)
                {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 
                    // cout<<"in"<<endl;
                    row_coo[next_empty_index] = i;
                    column_coo[next_empty_index] = j;
                    val[next_empty_index] = matrix[i][j];
                    next_empty_index ++;
                }
                
            }
        }
        return next_empty_index;
    }

    for(size_t batch = 0; batch< THREAD_NUM; ++batch)
    {
        threads_next_empty_index[batch][0] = next_empty_index;
        for (size_t i = first_last_index[batch][0]; i < first_last_index[batch][1]; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
            // cout<<"["<<i<<"]["<<j<<"]: "<<matrix[i][j]<<"\t"<<v[j]<<endl;

                if (matrix[i][j]!=0.0 && v[j] !=0.0)
                {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 
                    // cout<<"in"<<endl;
                    row_coo[next_empty_index] = i;
                    column_coo[next_empty_index] = j;
                    val[next_empty_index] = matrix[i][j];
                    next_empty_index ++;
                }
                
            }
        }

        threads_next_empty_index[batch][1] = next_empty_index;


    }

    

    /*
    cout<< "------------ coo format result ------------\n";
    for (size_t i = 0; i < next_empty_index; i++)
    {
        cout<<"("<<row_coo[i]<<", "<<column_coo[i]<<"): "<< val[i]<<endl;
    }
     
    */
    
    return -1;

}

void coo_threads_func(int first_index, int last_index)
{
    for (size_t i = first_index; i < last_index; ++i)
        result[row_coo[i]] += val[i] * v[column_coo[i]]; 

}


float coo_multi_parallel()
{

    int threads_next_empty[THREAD_NUM][2] {0};
    int next_empty_index = coo_constructor_parallel(threads_next_empty);
    
    
    auto start = high_resolution_clock::now();
    
    thread threads[THREAD_NUM]; 

    // if the row number of the matrix is atleast equals to number of thread, use multithread implementation,
    //  else calculate all at once with one thread (its too small that its faster to use one thread than dividing it to multiple ones) 
    if(next_empty_index > 0)
    {
        coo_threads_func(0, next_empty_index);
    }
    else
    {
        for (int i = 0; i < THREAD_NUM; ++i) 
        {   
            // cout<<"thread #: "<<i<< "\tstart: "<<first_row<<"\tend: "<<last_row<<endl;
            threads[i] = thread(coo_threads_func,threads_next_empty[i][0], threads_next_empty[i][1]);

        }

	for (int i = 0; i < THREAD_NUM; ++i) 
		threads[i].join();
    }


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();

}


/*
    ********************************************************************
    Parallel CSR implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/


void csr_threads_func(int first_empty_index, int first_row, int last_row)
{

    int upper_bound, skip_rows;

    for (size_t i = first_row; i < last_row; ++i)
    {
        
        skip_rows = 0; 
        // first skip empty rows from here, to find exactly where to stop considering columns values
        while(row_offset[i + skip_rows + 1] == -1)
            skip_rows ++;
        
        if(i + 1 + skip_rows == n)
            upper_bound = first_empty_index;
        else if (row_offset[i] == -1) // current row row is empty, skip it
        {
            continue; // go and run the loop for the next i 
        }
        
        else
            upper_bound = row_offset[i+ skip_rows + 1];
        
        
        for (size_t j = row_offset[i]; j < upper_bound; ++j)
        {
            result[i] += val[j]*v[columns[j]];
        }
        // jump to the first row that is not empty
        i+= skip_rows;
    }   

}

float csr_multi_parallel()
{

    // there is no different between this algorithm matrix format in parallel or single thread, use single thread function for construction appropriate matrix format
    int first_empty_index = csr_constructor();
    
    auto start = high_resolution_clock::now();
    int first_row, last_row, row_batch_size;
    
    thread threads[THREAD_NUM]; 

    // if the row number of the matrix is atleast equals to number of thread, use multithread implementation,
    //  else calculate all at once with one thread (its too small that its faster to use one thread than dividing it to multiple ones) 
	if(n < THREAD_NUM){
        first_row = 0;
        last_row = n;
        csr_threads_func(first_empty_index, first_row, last_row);
    }
    else
    {
        row_batch_size = n/THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; ++i) 
        {    
            switch (i)
            {
            case 0:
                first_row =0;
                last_row = row_batch_size;
                break;
            case THREAD_NUM:
                first_row = row_batch_size * (i);
                last_row = n;
                break;
            default:
                first_row = row_batch_size * (i);
                last_row = first_row + row_batch_size; 
                break;
            }

        // cout<<"thread #: "<<i<< "\tstart: "<<first_row<<"\tend: "<<last_row<<endl;
  		threads[i] = thread(csr_threads_func,first_empty_index, first_row, last_row);

        }

	for (int i = 0; i < THREAD_NUM; ++i) 
		threads[i].join();
    }


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();

}


/*
    ********************************************************************
    Parallel ELL implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/


void ell_threads_func(int max_row, int first_row, int last_row)
{
    int row_index;    
    for (size_t i = first_row; i < last_row; ++i)
    {
        for (size_t j = 0; j < max_row; ++j)
        {
            row_index = column_mat[i][j];
            if (row_index != -1)
            {
                result[i] += ell_val[i][j] * v[row_index];
            }
            else
            {   // the rest of this row is emtpy, go tho the next row 
                // (because we filled column_mat form left, till there is no row tho add)  
                break;
            }
        }
        
    }

}

float ell_multi_parallel()
{
    
    // there is no different between this algorithm matrix format in parallel or single thread, use single thread function for construction appropriate matrix format
    int max_row = ell_constructor(); 
    
    auto start = high_resolution_clock::now();
    
    int first_row,last_row, row_batch_size;
    thread threads[THREAD_NUM]; 



    // if the row number of the matrix is atleast equals to number of thread, use multithread implementation,
    //  else calculate all at once with one thread (its too small that its faster to use one thread than dividing it to multiple ones) 
	if(n < THREAD_NUM){
        first_row = 0;
        last_row = n;
        ell_threads_func(max_row, first_row, last_row);
    }
    else
    {
        row_batch_size = n/THREAD_NUM;
        for (int i = 0; i < THREAD_NUM; ++i) 
        {    
            switch (i)
            {
            case 0:
                first_row =0;
                last_row = row_batch_size;
                break;
            case THREAD_NUM:
                first_row = row_batch_size * (i);
                last_row = n;
                break;
            default:
                first_row = row_batch_size * (i);
                last_row = first_row + row_batch_size; 
                break;
        }

        // cout<<"thread #: "<<i<< "\tstart: "<<first_row<<"\tend: "<<last_row<<endl;
  		threads[i] = thread(ell_threads_func,max_row, first_row, last_row);

    }

	for (int i = 0; i < THREAD_NUM; ++i) 
		threads[i].join();
    }


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();

}




// fill(v, v+m, 0.0);
// fill(&matrix[0][0], &matrix[n][0]+(n*m), 0.0);


/* 
    ********************************************************************
                                SIMD
    ******************************************************************** 
*/

float sum8(__m128 hiQuad, __m128 loQuad) {

    // // hiQuad = ( x7, x6, x5, x4 )
    // const __m128 hiQuad = y;
    // // loQuad = ( x3, x2, x1, x0 )
    // const __m128 loQuad = x;
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


/*
    ********************************************************************
    SIMD CSR implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/

int  csr_constructor_simd()
{   

    fill(row_offset, row_offset + (max_dim+10), 0.0);
    fill(columns, columns + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(val, val + ((max_dim+10)*(max_dim+10)), 0.0);
    fill(result, result + (max_dim+10), 0.0);
    
    // first index of the empty cell in columns and val arrays
    int first_empty_index = 0,add_count;
    
    for (size_t i = 0; i < n; ++i)
    {   
        
        int starts_from = first_empty_index;
        for (size_t j = 0; j < m; ++j)
        {

            if(matrix[i][j] != 0.0 && v[j] != 0.0)
            {   // only consider those cell of the matrix that would have a nonzero product in matrix-vector multiplication 
        
                // store v[j] instead of j so we could use it to multiply faster 
                columns_simd[first_empty_index] = v[j];
                val_simd[first_empty_index] = matrix[i][j];
                // move the index to the first empty cell in the columns and val arrays to next cell 
                // (as they have been filled by one value)
                first_empty_index ++;

            }
        }

        // fill zero for the rest of column and values for this row till its 8k
        if((first_empty_index - starts_from)%8 > 0)
        {
            add_count  = 8 - (first_empty_index - starts_from)%8;
            for(int check =0; check < add_count; ++check)
            {
                columns_simd[first_empty_index] = 0;
                val_simd[first_empty_index] = 0;
                // move the index to the first empty cell in the columns and val arrays to next cell 
                // (as they have been filled by one value)
                first_empty_index ++;
            }
        }

        if(starts_from != first_empty_index) // this row is not empty
        {
            // add the starting point of first nonzero value of this row index in the columns and val array
            row_offset[i] = starts_from;
        }
        else
            row_offset[i] = -1;  // an invalid index to show that this row is empty
        
    }

    /*
    cout<< "------------ row offset ------------"<<endl;
    for (size_t i = 0; i < n; i++)
        cout<< row_offset[i]<< ' ';
    cout<< endl;

    cout<< "------------ columns ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< columns_simd[i]<< ' ';
    cout<< endl;

    cout<< "------------ values ------------"<<endl;
    for (size_t i = 0; i < first_empty_index; i++)
        cout<< val_simd[i]<< ' ';
    cout<< endl;
    */

    return first_empty_index;
}


float csr_multi_simd()
{
    int first_empty_index =  csr_constructor_simd(), upper_bound,skip_rows;
    
    auto start = high_resolution_clock::now();

    __m128 *vec = (__m128*)(columns_simd);
    __m128 *mat = (__m128*)(val_simd);
    for(int i =0; i < n; ++i)
    {
        
        if(row_offset[i] != -1)
        {
            __m128 acc_1 = _mm_setzero_ps();
            __m128 acc_2  = _mm_setzero_ps();
            skip_rows = 0; 
            // first skip empty rows from here, to find exactly where to stop considering columns values
            while(row_offset[i + skip_rows + 1] == -1)
                skip_rows ++;
            
            if(i + 1 + skip_rows == n)
                upper_bound = first_empty_index;
            else
                upper_bound = row_offset[i+ skip_rows + 1];
            
            for(int j =row_offset[i]/4; j< upper_bound/4; j+=2)
            {
                acc_1 = _mm_add_ps(acc_1,_mm_mul_ps(vec[j], mat[j]));

                acc_2 = _mm_add_ps(acc_2,_mm_mul_ps(vec[j+1], mat[j +1]));
            }
            result[i] = sum8(acc_1, acc_2);

        }
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    return (float)duration.count();


}


void run_all_algs()
{
    // n =20;
    // m = 4;
    
    n =max_dim;
    m = max_dim;

   
    generate_random_input();

    
    float coo_duration  = coo_multi();
    cout<< "coo duration: "<<coo_duration<<"  microseconds\n"<<endl;

    // print_initial_matrix();    
    // print_result();

    float csr_duration = csr_multi();
    cout<< "csr duration: "<<csr_duration<<"  microseconds\n"<<endl;

    // print_result();

    float csc_duration = csc_multi();
    cout<< "csc duration: "<<csc_duration<<"  microseconds\n"<<endl;

    // print_result();
   
    float ell_duration = ell_multi();
    cout<< "ell duration: "<<ell_duration<<"  microseconds\n"<<endl;

    // print_result();

    float coo_parallel_duration  = coo_multi_parallel();
    cout<< "coo parallel duration: "<<coo_parallel_duration<<"  microseconds\n"<<endl;

    // print_result();


    float csr_parallel_duration = csr_multi_parallel();
    cout<< "csr parallel duration: "<<csr_parallel_duration<<"  microseconds\n"<<endl;

    // print_result();
    
    float ell_parallel_duration = ell_multi_parallel();
    cout<< "ell parallel duration: "<<ell_parallel_duration<<"  microseconds\n"<<endl;


    // print_result();

    float csr_simd_duration = csr_multi_simd();
    cout<< "csr-simd duration: "<<csr_simd_duration<<"  microseconds\n"<<endl;

    // print_result();

}


/*
    ********************************************************************
    SIMD ELL implementation of Sparse matrix for faster SpMSpV multiplication
    ******************************************************************** 
*/



void evaluate_on_different_sparsity()
{

    string headers[6] {"alg", "microsec", "mspf", "vspf", "rn","cn"};
    
    n =max_dim;
    m = max_dim;

	fstream csv_file;
    // opens an existing csv file or creates a new file.
	try
	{
		remove("evaluation_data.csv");

	}
	catch(const exception& e)
	{
		std::cerr << e.what() << '\n';
	}
    csv_file.open("evaluation_data.csv", ios::out | ios::app);

    for(size_t i =0; i< 5; ++i)
    {
        csv_file<<headers[i]<<",";
    }
    csv_file<<headers[5]<<"\n";

    for(int v_factor = 10; v_factor< 100; v_factor +=10)
    {
        for(int m_factor = 50; m_factor< 100; m_factor+=10)
        {
            generate_random_input(m_factor,v_factor);
            cout<<"-- M-Sparsity: "<<m_factor<<"\tv-Sparsity: "<<v_factor<<endl;
            
            float coo_duration  = coo_multi();
            csv_file<<"coo"<<", "
                    <<coo_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";

            // cout<< "coo duration: "<<coo_duration<<"  microseconds\n"<<endl;

            float csr_duration = csr_multi();
            csv_file<<"csr"<<", "
                    <<csr_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";

            // cout<< "csr duration: "<<csr_duration<<"  microseconds\n"<<endl;

            float csc_duration = csc_multi();
            csv_file<<"csc"<<", "
                    <<csc_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // cout<< "csc duration: "<<csc_duration<<"  microseconds\n"<<endl;

            float ell_duration = ell_multi();
            csv_file<<"ell"<<", "
                    <<ell_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // cout<< "ell duration: "<<ell_duration<<"  microseconds\n"<<endl;

            float coo_parallel_duration  = coo_multi_parallel();
            csv_file<<"coo-pll"<<", "
                    <<coo_parallel_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // cout<< "coo-pll duration: "<<coo_parallel_duration<<"  microseconds\n"<<endl;

            float csr_parallel_duration = csr_multi_parallel();
            csv_file<<"csr-pll"<<", "
                    <<csr_parallel_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // cout<< "csr-pll duration: "<<csr_parallel_duration<<"  microseconds\n"<<endl;

            float ell_parallel_duration = ell_multi_parallel();
            csv_file<<"ell-pll"<<", "
                    <<ell_parallel_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // cout<< "ell-pll duration: "<<ell_parallel_duration<<"  microseconds\n"<<endl;

            float csr_simd_duration = csr_multi_simd();
            csv_file<<"csr-simd"<<", "
                    <<csr_simd_duration<<", "
                    <<m_factor<<", "
                    <<v_factor<< ", "
                    <<n<<", "
                    <<m<<"\n";
            // break;   
        }
        // break;
    }
}

int main()
{

    evaluate_on_different_sparsity();
    // run_all_algs();
    // n =max_dim;
    // m = max_dim;
    // cout<<size(columns_simd);
    

   
    // generate_random_input();
    // print_initial_matrix();
    
    // cout<<csr_multi_simd();
    // print_result();
    // // cout<<"\n";
    // csr_multi();
    // print_result();
    
    return 0;
}


//  auto start = high_resolution_clock::now();

//  auto stop = high_resolution_clock::now();
// 	auto duration = duration_cast<microseconds>(stop - start);

//     return (float)duration.count();
