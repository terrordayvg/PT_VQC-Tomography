# Create test_process_tomography_code for certain base functions
import process_tomography_code as po
  
def test_dec_to_bin(): #From decimal to binary N-bits string.
  # test 5 into 101
  # x=5, N=3
  x=5
  N=3
  res=po.dec_to_bin(x,N)
  assert res== [1,0,1]
  
