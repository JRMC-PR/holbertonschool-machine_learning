
## Databases

### Description
0. Create a databasemandatoryWrite a script that creates the databasedb_0in your MySQL server.If the databasedb_0already exists, your script should not failYou are not allowed to use theSELECTorSHOWstatementsguillaume@ubuntu:~/$ cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
Enter password: 
guillaume@ubuntu:~/$ echo "SHOW databases;" | mysql -hlocalhost -uroot -p
Enter password: 
Database
information_schema
db_0
mysql
performance_schema
guillaume@ubuntu:~/$ cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
Enter password: 
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:0-create_database_if_missing.sqlHelp×Students who are done with "0. Create a database"Review your work×Correction of "0. Create a database"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/6pts

1. First tablemandatoryWrite a script that creates a table calledfirst_tablein the current database in your MySQL server.first_tabledescription:idINTnameVARCHAR(256)The database name will be passed as an argument of themysqlcommandIf the tablefirst_tablealready exists, your script should not failYou are not allowed to use theSELECTorSHOWstatementsguillaume@ubuntu:~/$ cat 1-first_table.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ echo "SHOW TABLES;" | mysql -hlocalhost -uroot -p db_0
Enter password: 
Tables_in_db_0
first_table
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:1-first_table.sqlHelp×Students who are done with "1. First table"Review your work×Correction of "1. First table"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

2. List all in tablemandatoryWrite a script that lists all rows of the tablefirst_tablein your MySQL server.All fields should be printedThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:2-list_values.sqlHelp×Students who are done with "2. List all in table"Review your work×Correction of "2. List all in table"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

3. First addmandatoryWrite a script that inserts a new row in the tablefirst_tablein your MySQL server.New row:id=89name=Holberton SchoolThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
id  name
89  Holberton School
guillaume@ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
id  name
89  Holberton School
89  Holberton School
89  Holberton School
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:3-insert_value.sqlHelp×Students who are done with "3. First add"Review your work×Correction of "3. First add"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

4. Select the bestmandatoryWrite a script that lists all records with ascore >= 10in the tablesecond_tablein your MySQL server.Results should display both the score and the name (in this order)Records should be ordered by score (top first)The database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat setup.sql
-- Create table and insert data
CREATE TABLE IF NOT EXISTS second_table (
    id INT,
    name VARCHAR(256),
    score INT
);
INSERT INTO second_table (id, name, score) VALUES (1, "Bob", 14);
INSERT INTO second_table (id, name, score) VALUES (2, "Roy", 3);
INSERT INTO second_table (id, name, score) VALUES (3, "John", 10);
INSERT INTO second_table (id, name, score) VALUES (4, "Bryan", 8);

guillaume@ubuntu:~/$ cat setup.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ cat 4-best_score.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
score   name
14  Bob
10  John
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:4-best_score.sqlHelp×Students who are done with "4. Select the best"Review your work×Correction of "4. Select the best"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

5. AveragemandatoryWrite a script that computes the score average of all records in the tablesecond_tablein your MySQL server.The result column name should beaverageThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat setup.sql
-- Create table and insert data
CREATE TABLE IF NOT EXISTS second_table (
    id INT,
    name VARCHAR(256),
    score INT
);
INSERT INTO second_table (id, name, score) VALUES (1, "Bob", 14);
INSERT INTO second_table (id, name, score) VALUES (2, "Roy", 5);
INSERT INTO second_table (id, name, score) VALUES (3, "John", 10);
INSERT INTO second_table (id, name, score) VALUES (4, "Bryan", 8);

guillaume@ubuntu:~/$ cat setup.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
guillaume@ubuntu:~/$ cat 5-average.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
average
9.25
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:5-average.sqlHelp×Students who are done with "5. Average"Review your work×Correction of "5. Average"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

6. Temperatures #0mandatoryImport inhbtn_0c_0database this table dump:downloadWrite a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).guillaume@ubuntu:~/$ cat 6-avg_temperatures.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
Enter password: 
city    avg_temp
Chandler    72.8627
Gilbert 71.8088
Pismo beach 71.5147
San Francisco   71.4804
Sedona  70.7696
Phoenix 70.5882
Oakland 70.5637
Sunnyvale   70.5245
Chicago 70.4461
San Diego   70.1373
Glendale    70.1225
Sonoma  70.0392
Yuma    69.3873
San Jose    69.2990
Tucson  69.0245
Joliet  68.6716
Naperville  68.1029
Tempe   67.0441
Peoria  66.5392
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:6-avg_temperatures.sqlHelp×Students who are done with "6. Temperatures #0"Review your work×Correction of "6. Temperatures #0"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

7. Temperatures #2mandatoryImport inhbtn_0c_0database this table dump:download(same asTemperatures #0)Write a script that displays the max temperature of each state (ordered by State name).guillaume@ubuntu:~/$ cat 7-max_state.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
Enter password: 
state   max_temp
AZ  110
CA  110
IL  110
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:7-max_state.sqlHelp×Students who are done with "7. Temperatures #2"Review your work×Correction of "7. Temperatures #2"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

8. Genre ID by showmandatoryImport the database dump fromhbtn_0d_tvshowsto your MySQL server:downloadWrite a script that lists all shows contained inhbtn_0d_tvshowsthat have at least one genre linked.Each record should display:tv_shows.title-tv_show_genres.genre_idResults must be sorted in ascending order  bytv_shows.titleandtv_show_genres.genre_idYou can use only oneSELECTstatementThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 8-genre_id_by_show.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
title   genre_id
Breaking Bad    1
Breaking Bad    6
Breaking Bad    7
Breaking Bad    8
Dexter  1
Dexter  2
Dexter  6
Dexter  7
Dexter  8
Game of Thrones 1
Game of Thrones 3
Game of Thrones 4
House   1
House   2
New Girl    5
Silicon Valley  5
The Big Bang Theory 5
The Last Man on Earth   1
The Last Man on Earth   5
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:8-genre_id_by_show.sqlHelp×Students who are done with "8. Genre ID by show"Review your work×Correction of "8. Genre ID by show"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

9. No genremandatoryImport the database dump fromhbtn_0d_tvshowsto your MySQL server:downloadWrite a script that lists all shows contained inhbtn_0d_tvshowswithout a genre linked.Each record should display:tv_shows.title-tv_show_genres.genre_idResults must be sorted in ascending order bytv_shows.titleandtv_show_genres.genre_idYou can use only oneSELECTstatementThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 9-no_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
title   genre_id
Better Call Saul    NULL
Homeland    NULL
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:9-no_genre.sqlHelp×Students who are done with "9. No genre"Review your work×Correction of "9. No genre"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

10. Number of shows by genremandatoryImport the database dump fromhbtn_0d_tvshowsto your MySQL server:downloadWrite a script that lists all genres fromhbtn_0d_tvshowsand displays the number of shows linked to each.Each record should display:<TV Show genre>-<Number of shows linked to this genre>First column must be calledgenreSecond column must be callednumber_of_showsDon’t display a genre that doesn’t have any shows linkedResults must be sorted in descending order by the number of shows linkedYou can use only oneSELECTstatementThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 10-count_shows_by_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
genre   number_of_shows
Drama   5
Comedy  4
Mystery 2
Crime   2
Suspense    2
Thriller    2
Adventure   1
Fantasy 1
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:10-count_shows_by_genre.sqlHelp×Students who are done with "10. Number of shows by genre"Review your work×Correction of "10. Number of shows by genre"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

11. Rotten tomatoesmandatoryImport the databasehbtn_0d_tvshows_ratedump to your MySQL server:downloadWrite a script that lists all shows fromhbtn_0d_tvshows_rateby their rating.Each record should display:tv_shows.title-rating sumResults must be sorted in descending order by the ratingYou can use only oneSELECTstatementThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 11-rating_shows.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
Enter password: 
title   rating
Better Call Saul    163
Homeland    145
Silicon Valley  82
Game of Thrones 79
Dexter  24
House   21
Breaking Bad    16
The Last Man on Earth   10
The Big Bang Theory 0
New Girl    0
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:11-rating_shows.sqlHelp×Students who are done with "11. Rotten tomatoes"Review your work×Correction of "11. Rotten tomatoes"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

12. Best genremandatoryImport the database dump fromhbtn_0d_tvshows_rateto your MySQL server:downloadWrite a script that lists all genres in the databasehbtn_0d_tvshows_rateby their rating.Each record should display:tv_genres.name-rating sumResults must be sorted in descending order by their ratingYou can use only oneSELECTstatementThe database name will be passed as an argument of themysqlcommandguillaume@ubuntu:~/$ cat 12-rating_genres.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
Enter password: 
name    rating
Drama   150
Comedy  92
Adventure   79
Fantasy 79
Mystery 45
Crime   40
Suspense    40
Thriller    40
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:12-rating_genres.sqlHelp×Students who are done with "12. Best genre"Review your work×Correction of "12. Best genre"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/6pts

13. We are all unique!mandatoryWrite a SQL script that creates a tableusersfollowing these requirements:With these attributes:id, integer, never null, auto increment and primary keyemail, string (255 characters), never null and uniquename, string (255 characters)If the table already exists, your script should not failYour script can be executed on any databaseContext:Make an attribute unique directly in the table schema will enforced your business rules and avoid bugs in your applicationbob@dylan:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn't exist
bob@dylan:~$ 
bob@dylan:~$ cat 13-uniq_users.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");' | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ echo 'INSERT INTO users (email, name) VALUES ("sylvie@dylan.com", "Sylvie");' | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Jean");' | mysql -uroot -p holberton
Enter password: 
ERROR 1062 (23000) at line 1: Duplicate entry 'bob@dylan.com' for key 'email'
bob@dylan:~$ 
bob@dylan:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
id  email   name
1   bob@dylan.com   Bob
2   sylvie@dylan.com    Sylvie
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:13-uniq_users.sqlHelp×Students who are done with "13. We are all unique!"Review your work×Correction of "13. We are all unique!"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/8pts

14. In and not outmandatoryWrite a SQL script that creates a tableusersfollowing these requirements:With these attributes:id, integer, never null, auto increment and primary keyemail, string (255 characters), never null and uniquename, string (255 characters)country, enumeration of countries:US,COandTN, never null (= default will be the first element of the enumeration, hereUS)If the table already exists, your script should not failYour script can be executed on any databasebob@dylan:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn't exist
bob@dylan:~$ 
bob@dylan:~$ cat 14-country_users.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ echo 'INSERT INTO users (email, name, country) VALUES ("bob@dylan.com", "Bob", "US");' | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ echo 'INSERT INTO users (email, name, country) VALUES ("sylvie@dylan.com", "Sylvie", "CO");' | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ echo 'INSERT INTO users (email, name, country) VALUES ("jean@dylan.com", "Jean", "FR");' | mysql -uroot -p holberton
Enter password: 
ERROR 1265 (01000) at line 1: Data truncated for column 'country' at row 1
bob@dylan:~$ 
bob@dylan:~$ echo 'INSERT INTO users (email, name) VALUES ("john@dylan.com", "John");' | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
id  email   name    country
1   bob@dylan.com   Bob US
2   sylvie@dylan.com    Sylvie  CO
3   john@dylan.com  John    US
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:14-country_users.sqlHelp×Students who are done with "14. In and not out"Review your work×Correction of "14. In and not out"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/7pts

15. Best band ever!mandatoryWrite a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fansRequirements:Import this table dump:metal_bands.sql.zipColumn names must be:originandnb_fansYour script can be executed on any databaseContext:Calculate/compute something is always power intensive… better to distribute the load!bob@dylan:~$ cat metal_bands.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 15-fans.sql | mysql -uroot -p holberton > tmp_res ; head tmp_res
Enter password: 
origin  nb_fans
USA 99349
Sweden  47169
Finland 32878
United Kingdom  32518
Germany 29486
Norway  22405
Canada  8874
The Netherlands 8819
Italy   7178
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:15-fans.sqlHelp×Students who are done with "15. Best band ever!"Review your work×Correction of "15. Best band ever!"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

16. Old school bandmandatoryWrite a SQL script that lists all bands withGlam rockas their main style, ranked by their longevityRequirements:Import this table dump:metal_bands.sql.zipColumn names must be:band_namelifespanuntil 2020 (in years)You should use attributesformedandsplitfor computing thelifespanYour script can be executed on any databasebob@dylan:~$ cat metal_bands.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 16-glam_rock.sql | mysql -uroot -p holberton 
Enter password: 
band_name   lifespan
Alice Cooper    56
Mötley Crüe   34
Marilyn Manson  31
The 69 Eyes 30
Hardcore Superstar  23
Nasty Idols 0
Hanoi Rocks 0
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:16-glam_rock.sqlHelp×Students who are done with "16. Old school band"Review your work×Correction of "16. Old school band"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

17. Buy buy buymandatoryWrite a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.Quantity in the tableitemscan be negative.Context:Updating multiple tables for one action from your application can generate issue: network disconnection, crash, etc… to keep your data in a good shape, let MySQL do it for you!bob@dylan:~$ cat 17-init.sql
-- Initial
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS orders;

CREATE TABLE IF NOT EXISTS items (
    name VARCHAR(255) NOT NULL,
    quantity int NOT NULL DEFAULT 10
);

CREATE TABLE IF NOT EXISTS orders (
    item_name VARCHAR(255) NOT NULL,
    number int NOT NULL
);

INSERT INTO items (name) VALUES ("apple"), ("pineapple"), ("pear");

bob@dylan:~$ 
bob@dylan:~$ cat 17-init.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 17-store.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 17-main.sql
Enter password: 
-- Show and add orders
SELECT * FROM items;
SELECT * FROM orders;

INSERT INTO orders (item_name, number) VALUES ('apple', 1);
INSERT INTO orders (item_name, number) VALUES ('apple', 3);
INSERT INTO orders (item_name, number) VALUES ('pear', 2);

SELECT "--";

SELECT * FROM items;
SELECT * FROM orders;

bob@dylan:~$ 
bob@dylan:~$ cat 17-main.sql | mysql -uroot -p holberton 
Enter password: 
name    quantity
apple   10
pineapple   10
pear    10
--
--
name    quantity
apple   6
pineapple   10
pear    8
item_name   number
apple   1
apple   3
pear    2
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:17-store.sqlHelp×Students who are done with "17. Buy buy buy"Review your work×Correction of "17. Buy buy buy"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

18. Email validation to sentmandatoryWrite a SQL script that creates a trigger that resets the attributevalid_emailonly when theemailhas been changed.Context:Nothing related to MySQL, but perfect for user email validation - distribute the logic to the database itself!bob@dylan:~$ cat 18-init.sql
-- Initial
DROP TABLE IF EXISTS users;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    email varchar(255) not null,
    name varchar(255),
    valid_email boolean not null default 0,
    PRIMARY KEY (id)
);

INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");
INSERT INTO users (email, name, valid_email) VALUES ("sylvie@dylan.com", "Sylvie", 1);
INSERT INTO users (email, name, valid_email) VALUES ("jeanne@dylan.com", "Jeanne", 1);

bob@dylan:~$ 
bob@dylan:~$ cat 18-init.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 18-valid_email.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 18-main.sql
Enter password: 
-- Show users and update (or not) email
SELECT * FROM users;

UPDATE users SET valid_email = 1 WHERE email = "bob@dylan.com";
UPDATE users SET email = "sylvie+new@dylan.com" WHERE email = "sylvie@dylan.com";
UPDATE users SET name = "Jannis" WHERE email = "jeanne@dylan.com";

SELECT "--";
SELECT * FROM users;

UPDATE users SET email = "bob@dylan.com" WHERE email = "bob@dylan.com";

SELECT "--";
SELECT * FROM users;

bob@dylan:~$ 
bob@dylan:~$ cat 18-main.sql | mysql -uroot -p holberton 
Enter password: 
id  email   name    valid_email
1   bob@dylan.com   Bob 0
2   sylvie@dylan.com    Sylvie  1
3   jeanne@dylan.com    Jeanne  1
--
--
id  email   name    valid_email
1   bob@dylan.com   Bob 1
2   sylvie+new@dylan.com    Sylvie  0
3   jeanne@dylan.com    Jannis  1
--
--
id  email   name    valid_email
1   bob@dylan.com   Bob 1
2   sylvie+new@dylan.com    Sylvie  0
3   jeanne@dylan.com    Jannis  1
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:18-valid_email.sqlHelp×Students who are done with "18. Email validation to sent"Review your work×Correction of "18. Email validation to sent"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

19. Add bonusmandatoryWrite a SQL script that creates a stored procedureAddBonusthat adds a new correction for a student.Requirements:ProcedureAddBonusis taking 3 inputs (in this order):user_id, ausers.idvalue (you can assumeuser_idis linked to an existingusers)project_name, a new or already existsprojects- if noprojects.namefound in the table, you should create itscore, the score value for the correctionContext:Write code in SQL is a nice level up!bob@dylan:~$ cat 19-init.sql
-- Initial
DROP TABLE IF EXISTS corrections;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    average_score float default 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS projects (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS corrections (
    user_id int not null,
    project_id int not null,
    score int default 0,
    KEY `user_id` (`user_id`),
    KEY `project_id` (`project_id`),
    CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
);

INSERT INTO users (name) VALUES ("Bob");
SET @user_bob = LAST_INSERT_ID();

INSERT INTO users (name) VALUES ("Jeanne");
SET @user_jeanne = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("C is fun");
SET @project_c = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("Python is cool");
SET @project_py = LAST_INSERT_ID();


INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_c, 80);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_py, 96);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_c, 91);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_py, 73);

bob@dylan:~$ 
bob@dylan:~$ cat 19-init.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 19-bonus.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 19-main.sql
Enter password: 
-- Show and add bonus correction
SELECT * FROM projects;
SELECT * FROM corrections;

SELECT "--";

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "Python is cool", 100);

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "Bonus project", 100);
CALL AddBonus((SELECT id FROM users WHERE name = "Bob"), "Bonus project", 10);

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "New bonus", 90);

SELECT "--";

SELECT * FROM projects;
SELECT * FROM corrections;

bob@dylan:~$ 
bob@dylan:~$ cat 19-main.sql | mysql -uroot -p holberton 
Enter password: 
id  name
1   C is fun
2   Python is cool
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
--
--
--
--
id  name
1   C is fun
2   Python is cool
3   Bonus project
4   New bonus
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
2   2   100
2   3   100
1   3   10
2   4   90
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:19-bonus.sqlHelp×Students who are done with "19. Add bonus"Review your work×Correction of "19. Add bonus"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

20. Average scoremandatoryWrite a SQL script that creates a stored procedureComputeAverageScoreForUserthat computes and store the average score for a student.Requirements:ProcedureComputeAverageScoreForUseris taking 1 input:user_id, ausers.idvalue (you can assumeuser_idis linked to an existingusers)bob@dylan:~$ cat 20-init.sql
-- Initial
DROP TABLE IF EXISTS corrections;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    average_score float default 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS projects (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS corrections (
    user_id int not null,
    project_id int not null,
    score int default 0,
    KEY `user_id` (`user_id`),
    KEY `project_id` (`project_id`),
    CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
);

INSERT INTO users (name) VALUES ("Bob");
SET @user_bob = LAST_INSERT_ID();

INSERT INTO users (name) VALUES ("Jeanne");
SET @user_jeanne = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("C is fun");
SET @project_c = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("Python is cool");
SET @project_py = LAST_INSERT_ID();


INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_c, 80);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_py, 96);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_c, 91);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_py, 73);

bob@dylan:~$ 
bob@dylan:~$ cat 20-init.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 20-average_score.sql | mysql -uroot -p holberton 
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 20-main.sql
-- Show and compute average score
SELECT * FROM users;
SELECT * FROM corrections;

SELECT "--";
CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = "Jeanne"));

SELECT "--";
SELECT * FROM users;

bob@dylan:~$ 
bob@dylan:~$ cat 20-main.sql | mysql -uroot -p holberton 
Enter password: 
id  name    average_score
1   Bob 0
2   Jeanne  0
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
--
--
--
--
id  name    average_score
1   Bob 0
2   Jeanne  82
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:20-average_score.sqlHelp×Students who are done with "20. Average score"Review your work×Correction of "20. Average score"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

21. Safe dividemandatoryWrite a SQL script that creates a functionSafeDivthat divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.Requirements:You must create a functionThe functionSafeDivtakes 2 arguments:a, INTb, INTAnd returnsa / bor 0 ifb == 0bob@dylan:~$ cat 21-init.sql
-- Initial
DROP TABLE IF EXISTS numbers;

CREATE TABLE IF NOT EXISTS numbers (
    a int default 0,
    b int default 0
);

INSERT INTO numbers (a, b) VALUES (10, 2);
INSERT INTO numbers (a, b) VALUES (4, 5);
INSERT INTO numbers (a, b) VALUES (2, 3);
INSERT INTO numbers (a, b) VALUES (6, 3);
INSERT INTO numbers (a, b) VALUES (7, 0);
INSERT INTO numbers (a, b) VALUES (6, 8);

bob@dylan:~$ cat 21-init.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ cat 21-div.sql | mysql -uroot -p holberton
Enter password: 
bob@dylan:~$ 
bob@dylan:~$ echo "SELECT (a / b) FROM numbers;" | mysql -uroot -p holberton
Enter password: 
(a / b)
5.0000
0.8000
0.6667
2.0000
NULL
0.7500
bob@dylan:~$ 
bob@dylan:~$ echo "SELECT SafeDiv(a, b) FROM numbers;" | mysql -uroot -p holberton
Enter password: 
SafeDiv(a, b)
5
0.800000011920929
0.6666666865348816
2
0
0.75
bob@dylan:~$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:21-div.sqlHelp×Students who are done with "21. Safe divide"Review your work×Correction of "21. Safe divide"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/4pts

22. List all databasesmandatoryWrite a script that lists all databases in MongoDB.guillaume@ubuntu:~/$ cat 22-list_databases | mongo
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017
MongoDB server version: 4.4.29
admin        0.000GB
config       0.000GB
local        0.000GB
logs         0.005GB
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:22-list_databasesHelp×Students who are done with "22. List all databases"Review your work×Correction of "22. List all databases"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/9pts

23. Create a databasemandatoryWrite a script that creates or uses the databasemy_db:guillaume@ubuntu:~/$ cat 22-list_databases | mongo
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017
MongoDB server version: 4.4.29
admin        0.000GB
config       0.000GB
local        0.000GB
logs         0.005GB
bye
guillaume@ubuntu:~/$
guillaume@ubuntu:~/$ cat 23-use_or_create_database | mongo
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017
MongoDB server version: 4.4.29
switched to db my_db
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:23-use_or_create_databaseHelp×Students who are done with "23. Create a database"Review your work×Correction of "23. Create a database"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/8pts

24. Insert documentmandatoryWrite a script that inserts a document in the collectionschool:The document must have one attributenamewith value “Holberton school”The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 24-insert | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
WriteResult({ "nInserted" : 1 })
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:24-insertHelp×Students who are done with "24. Insert document"Review your work×Correction of "24. Insert document"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/8pts

25. All documentsmandatoryWrite a script that lists all documents in the collectionschool:The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 25-all | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
{ "_id" : ObjectId("5a8fad532b69437b63252406"), "name" : "Holberton school" }
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:25-allHelp×Students who are done with "25. All documents"Review your work×Correction of "25. All documents"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/9pts

26. All matchesmandatoryWrite a script that lists all documents withname="Holberton school"in the collectionschool:The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 26-match | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
{ "_id" : ObjectId("5a8fad532b69437b63252406"), "name" : "Holberton school" }
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:26-matchHelp×Students who are done with "26. All matches"Review your work×Correction of "26. All matches"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/11pts

27. CountmandatoryWrite a script that displays the number of documents in the collectionschool:The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 27-count | mongo my_db
MongoDB shell version v4.4.2
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.2
1
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:27-countHelp×Students who are done with "27. Count"Review your work×Correction of "27. Count"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/9pts

28. UpdatemandatoryWrite a script that adds a new attribute to a document in the collectionschool:The script should update only document withname="Holberton school"(all of them)The update should add the attributeaddresswith the value “972 Mission street”The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 28-update | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
WriteResult({ "nMatched" : 1, "nUpserted" : 0, "nModified" : 1 })
bye
guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ cat 26-match | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
{ "_id" : ObjectId("5a8fad532b69437b63252406"), "name" : "Holberton school", "address" : "972 Mission street" }
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:28-updateHelp×Students who are done with "28. Update"Review your work×Correction of "28. Update"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/11pts

29. Delete by matchmandatoryWrite a script that deletes all documents withname="Holberton school"in the collectionschool:The database name will be passed as option ofmongocommandguillaume@ubuntu:~/$ cat 29-delete | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
{ "acknowledged" : true, "deletedCount" : 1 }
bye
guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ cat 26-match | mongo my_db
MongoDB shell version v4.4.29
connecting to: mongodb://127.0.0.1:27017/my_db
MongoDB server version: 4.4.29
bye
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:29-deleteHelp×Students who are done with "29. Delete by match"Review your work×Correction of "29. Delete by match"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/11pts

30. List all documents in PythonmandatoryWrite a Python function that lists all documents in a collection:Prototype:def list_all(mongo_collection):Return an empty list if no document in the collectionmongo_collectionwill be thepymongocollection objectguillaume@ubuntu:~/$ cat 30-main.py
#!/usr/bin/env python3
""" 30-main """
from pymongo import MongoClient
list_all = __import__('30-all').list_all

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {}".format(school.get('_id'), school.get('name')))

guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ ./30-main.py
[5a8f60cfd4321e1403ba7ab9] Holberton school
[5a8f60cfd4321e1403ba7aba] UCSD
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:30-all.pyHelp×Students who are done with "30. List all documents in Python"Review your work×Correction of "30. List all documents in Python"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/9pts

31. Insert a document in PythonmandatoryWrite a Python function that inserts a new document in a collection based onkwargs:Prototype:def insert_school(mongo_collection, **kwargs):mongo_collectionwill be thepymongocollection objectReturns the new_idguillaume@ubuntu:~/$ cat 31-main.py
#!/usr/bin/env python3
""" 31-main """
from pymongo import MongoClient
list_all = __import__('30-all').list_all
insert_school = __import__('31-insert_school').insert_school

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    new_school_id = insert_school(school_collection, name="UCSF", address="505 Parnassus Ave")
    print("New school created: {}".format(new_school_id))

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'), school.get('name'), school.get('address', "")))

guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ ./31-main.py
New school created: 5a8f60cfd4321e1403ba7abb
[5a8f60cfd4321e1403ba7ab9] Holberton school
[5a8f60cfd4321e1403ba7aba] UCSD
[5a8f60cfd4321e1403ba7abb] UCSF 505 Parnassus Ave
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:31-insert_school.pyHelp×Students who are done with "31. Insert a document in Python"Review your work×Correction of "31. Insert a document in Python"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/14pts

32. Change school topicsmandatoryWrite a Python function that changes all topics of a school document based on the name:Prototype:def update_topics(mongo_collection, name, topics):mongo_collectionwill be thepymongocollection objectname(string) will be the school name to updatetopics(list of strings) will be the list of topics approached in the schoolguillaume@ubuntu:~/$ cat 32-main.py
#!/usr/bin/env python3
""" 32-main """
from pymongo import MongoClient
list_all = __import__('30-all').list_all
update_topics = __import__('32-update_topics').update_topics

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    update_topics(school_collection, "Holberton school", ["Sys admin", "AI", "Algorithm"])

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'), school.get('name'), school.get('topics', "")))

    update_topics(school_collection, "Holberton school", ["iOS"])

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'), school.get('name'), school.get('topics', "")))

guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ ./32-main.py
[5a8f60cfd4321e1403ba7abb] UCSF 
[5a8f60cfd4321e1403ba7aba] UCSD 
[5a8f60cfd4321e1403ba7ab9] Holberton school ['Sys admin', 'AI', 'Algorithm']
[5a8f60cfd4321e1403ba7abb] UCSF 
[5a8f60cfd4321e1403ba7aba] UCSD 
[5a8f60cfd4321e1403ba7ab9] Holberton school ['iOS']
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:32-update_topics.pyHelp×Students who are done with "32. Change school topics"Review your work×Correction of "32. Change school topics"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/11pts

33. Where can I learn Python?mandatoryWrite a Python function that returns the list of school having a specific topic:Prototype:def schools_by_topic(mongo_collection, topic):mongo_collectionwill be thepymongocollection objecttopic(string) will be topic searchedguillaume@ubuntu:~/$ cat 33-main.py
#!/usr/bin/env python3
""" 33-main """
from pymongo import MongoClient
list_all = __import__('30-all').list_all
insert_school = __import__('31-insert_school').insert_school
schools_by_topic = __import__('33-schools_by_topic').schools_by_topic

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school

    j_schools = [
        { 'name': "Holberton school", 'topics': ["Algo", "C", "Python", "React"]},
        { 'name': "UCSF", 'topics': ["Algo", "MongoDB"]},
        { 'name': "UCLA", 'topics': ["C", "Python"]},
        { 'name': "UCSD", 'topics': ["Cassandra"]},
        { 'name': "Stanford", 'topics': ["C", "React", "Javascript"]}
    ]
    for j_school in j_schools:
        insert_school(school_collection, **j_school)

    schools = schools_by_topic(school_collection, "Python")
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'), school.get('name'), school.get('topics', "")))

guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ ./33-main.py
[5a90731fd4321e1e5a3f53e3] Holberton school ['Algo', 'C', 'Python', 'React']
[5a90731fd4321e1e5a3f53e5] UCLA ['C', 'Python']
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:33-schools_by_topic.pyHelp×Students who are done with "33. Where can I learn Python?"Review your work×Correction of "33. Where can I learn Python?"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/13pts

34. Log statsmandatoryWrite a Python script that provides some stats about Nginx logs stored in MongoDB:Database:logsCollection:nginxDisplay (same as the example):first line:x logswherexis the number of documents in this collectionsecond line:Methods:5 lines with the number of documents with themethod=["GET", "POST", "PUT", "PATCH", "DELETE"]in this order (see example below - warning: it’s a tabulation before each line)one line with the number of documents with:method=GETpath=/statusYou can use this dump as data sample:dump.zipThe output of your scriptmust be exactly the same as the exampleguillaume@ubuntu:~/$ curl -o dump.zip -s "https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-webstack/411/dump.zip"
guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ unzip dump.zip
Archive:  dump.zip
   creating: dump/
   creating: dump/logs/
  inflating: dump/logs/nginx.metadata.json
  inflating: dump/logs/nginx.bson
guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ mongorestore dump
2024-09-03T13:26:38.139+0000    preparing collections to restore from
2024-09-03T13:26:38.140+0000    reading metadata for logs.nginx from dump/logs/nginx.metadata.json
2024-09-03T13:26:47.938+0000    restoring logs.nginx from dump/logs/nginx.bson
2024-09-03T13:26:49.014+0000    [#######################.]  logs.nginx  13.0MB/13.4MB  (96.9%)
2024-09-03T13:26:49.043+0000    [########################]  logs.nginx  13.4MB/13.4MB  (100.0%)
2024-09-03T13:26:49.043+0000    finished restoring logs.nginx (94778 documents, 0 failures)
2024-09-03T13:26:49.043+0000    no indexes to restore for collection logs.nginx
2024-09-03T13:26:49.043+0000    94778 document(s) restored successfully. 0 document(s) failed to restore.
guillaume@ubuntu:~/$ 
guillaume@ubuntu:~/$ ./34-log_stats.py 
94778 logs
Methods:
        method GET: 93842
        method POST: 229
        method PUT: 0
        method PATCH: 0
        method DELETE: 0
47415 status check
guillaume@ubuntu:~/$Repo:GitHub repository:holbertonschool-machine_learningDirectory:pipeline/databasesFile:34-log_stats.pyHelp×Students who are done with "34. Log stats"Review your work×Correction of "34. Log stats"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failedGet a sandbox0/12pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Databases.md`
